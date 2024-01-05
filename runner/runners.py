from .base import Trainer, Visualizer
import torch
from src.utils import load_model,get_attitude, get_attitude_score, clear_gpu_memory, get_gpu_memory, clean_and_link_checkpoints
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
from transformers import pipeline

class SentimentClassificationTrainer(Trainer):
    def __init__(self, model, data, conf) -> None:
        super().__init__(model, data, conf)

    def prepare_para(self):
        pass

    def preprocess_data(self, batch, with_answer=False):
        if with_answer:
            for i in range(len(batch[self.x])):
                ans = f" (It is {'positive' if batch[self.y][i] == 1 else 'negative'}.)"
                batch[self.x][i] = batch[self.x][i] + ans + " --- Is this comment positive or negative?"
            batch_texts = batch[self.x]
        else:
            batch_texts = [data + " --- Is this comment positive or negative?" for data in batch[self.x]]
        inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        inputs.to(self.device)
        return inputs.input_ids

    def inference(self, model, x):
        ans = model.generate(x, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
        # feature_extractor = pipeline(task="feature-extraction", tokenizer=self.tokenizer, model=model, device=self.device)
        # feature = torch.tensor(feature_extractor(x))
        return torch.stack(ans.scores).permute(1,0,2), ans.sequences

    def get_text(self, sequences_ids):
        decoded_sequences = [self.tokenizer.decode(output, skip_special_tokens=False) for output in sequences_ids]
        return decoded_sequences

    def run(self):
        self.set_seed(31415926)
        self.llm.train()
        self.step = 0
        if self.load_from:
            load_model(self.load_from, self)
        for i in range(self.epochs):
            for batch in tqdm(self.dataloader):
                # try:
                inp = self.preprocess_data(batch)
                inp_with_ans = self.preprocess_data(batch, with_answer=True)
                benchmark, benchmark_ids = self.inference(self.ori_llm, inp_with_ans)
                benchmark_seqs = self.get_text(benchmark_ids)

                self.logger.log("\n---------------------------------------------")
                self.logger.log("Benchmark: ")
                for benchmark_seq in benchmark_seqs:
                    self.logger.log(benchmark_seq)
                self.logger.log("--------------------------------------------- \n")

                sample_features = []
                loss_1 = 0.0
                loss_2 = 0.0

                for j in range(10):  # samples number
                    sample_feature, one_sample_ids = self.inference(self.llm, inp)

                    # Get the sample sequence and print them out
                    samples_seqs = self.get_text(one_sample_ids)
                    self.logger.log(f"Sample {j + 1}: ")
                    for sample_seq in samples_seqs:
                        self.logger.log(sample_seq)

                    sample_features.append(sample_feature)
                    loss_1 += torch.dist(self.feature_map()['rec'], self.feature_map()['ori'], p=2)
                    loss_2 += torch.dist(sample_feature, benchmark, p=2)
                    loss = (1 - self.lambda_) * loss_1 + self.lambda_ * loss_2

                    loss /= 10
                    loss_1 /= 10
                    loss_2 /= 10

                    self.logger.log(f"ITRATION: {self.step}")
                    self.logger.log(f"LAMBDA: {self.lambda_}")
                    self.logger.log(f"LOSS: {loss}")
                    self.logger.log(f"LOSS 1: {loss_1}")
                    self.logger.log(f"LOSS 2: {loss_2}")
                    self.writer_.add_scalar("Loss 1", loss_1, self.step)
                    self.writer_.add_scalar("Loss 2", loss_2, self.step)
                    self.writer_.add_scalar("Loss", loss, self.step)
                    self.step += 1
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    self.lambda_ = self.lambda_scheduler.update(self.step, 1550)

                    if self.step % 50 == 0:
                        checkpoint = {
                            'epoch': i + 1,
                            'step': self.step + 1,
                            'model_state_dict': self.llm.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss
                        }
                        torch.save(checkpoint, self.checkpoints_path + f"/epoch-{i + 1}_step-{self.step}.pth")
                # except RuntimeError as e:
                #     if 'out of memory' in str(e):
                #         print('| WARNING: ran out of memory, skipping batch')
                #         self.optimizer.zero_grad()
                #         torch.cuda.empty_cache()
                #     else:
                #         raise e

            self.scheduler.step()


class FeatureMapVisiualizeRunner(Visualizer):
    def __init__(self, model, data, conf) -> None:
        super().__init__(model, data, conf)
        self.fms = {}

    def inference(self, model, x):
        ans = model.generate(x, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
        return torch.stack(ans.scores).permute(1,0,2), ans.sequences

    def get_text(self, sequences_ids):
        decoded_sequences = [self.tokenizer.decode(output, skip_special_tokens=False) for output in sequences_ids]
        return decoded_sequences

    def preprocess_data(self, batch, with_answer=False):
        if with_answer:
            for i in range(len(batch[self.x])):
                ans = f" (It is {'positive' if batch[self.y][i] == 1 else 'negative'}.)"
                batch[self.x][i] = batch[self.x][i] + ans + " --- Is this comment positive or negative?"
            batch_texts = batch[self.x]
        else:
            batch_texts = [data + " --- Is this comment positive or negative?" for data in batch[self.x]]
        inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        inputs.to(self.device)
        return inputs.input_ids

    def draw(self, data, title, path_):
        sliced_tensors = [data[0, i, :].cpu() for i in range(data.size(1))]
        for i, label in enumerate(title):
            try:
                data_2d = sliced_tensors[i].view(64, 64).numpy()
            except:
                data_2d = sliced_tensors[i].view(64, 64).detach().numpy()
            self.fms[label] = data_2d
            plt.figure(figsize=(6, 5))
            plt.imshow(data_2d, cmap='viridis')
            plt.title(label)
            plt.colorbar()
            plt.savefig(os.path.join(path_, f"{label}_sum{np.sum(data_2d)}.png"))
            plt.close()

    def run(self):
        self.set_seed(31415926)
        self.llm.eval()
        self.bs_num = 0
        if self.load_from:
            load_model(self.load_from, self)
        for idx, batch in enumerate(tqdm(self.dataloader)):
            inp = self.preprocess_data(batch)
            inp_with_ans = self.preprocess_data(batch, with_answer=True)
            benchmark, benchmark_ids = self.inference(self.ori_llm, inp_with_ans)
            benchmark_seqs = self.get_text(benchmark_ids)
            self.logger.log("\n-------------------------------------")
            for ii, benchmark_seq in enumerate(benchmark_seqs):
                    self.logger.log(f"{ii}. {benchmark_seq}")
            self.logger.log("**********************************\n")
            benchmark_plot_names = [f"bm_{benchmark_seqs[x][0:10]}_batch{idx}_idx{x}" for x in range(len(benchmark_seqs))]
            for i in range(10):
                sample_feature, sample_ids = self.inference(self.llm, inp)
                samples_seqs = self.get_text(sample_ids)
                for ii, sample_seq in enumerate(samples_seqs):
                    self.logger.log(f"{ii}. {sample_seq}")
                self.logger.log("-------------------------------------")
                featuremap_plot_names = [f"fm_{samples_seqs[x][0:10]}_batch{idx}_idx{x}_it{i}" for x in range(len(samples_seqs))]
                if i == 0:
                    self.draw(self.feature_map()['ori'], benchmark_plot_names, self.run_path)
                self.draw(self.feature_map()['rec'], featuremap_plot_names, self.run_path)

            self.bs_num += 1
            if self.bs_num >= self.iteration:
                break


class ValidateAttitudeRunner(Visualizer):
    def __init__(self, model, data, conf) -> None:
        super().__init__(model, data, conf)

    def inference(self, model, x):
        ans = model.generate(x, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
        return torch.stack(ans.scores).permute(1,0,2), ans.sequences

    def get_text(self, sequences_ids):
        decoded_sequences = [self.tokenizer.decode(output, skip_special_tokens=False) for output in sequences_ids]
        return decoded_sequences

    def get_answer_str(self, squences_list):
        tmp = [x.split("Is this comment positive or negative?") for x in squences_list]
        return [x[1].strip() if len(x) > 1 else "" for x in tmp]

    def get_answer_attitude(self):
        pass

    def preprocess_data(self, batch, with_answer=False):
        if with_answer:
            for i in range(len(batch[self.x])):
                ans = f" (It is {'positive' if batch[self.y][i] == 1 else 'negative'}.)"
                batch[self.x][i] = batch[self.x][i] + " --- Is this comment positive or negative?" + ans
            batch_texts = batch[self.x]
        else:
            batch_texts = [data + " --- Is this comment positive or negative?" for data in batch[self.x]]
        inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        inputs.to(self.device)
        return inputs.input_ids

    def run(self):
        self.set_seed(31415926)
        self.llm.eval()
        self.bs_num = 0
        self.sample_answers = []
        right_num = 0
        sample_num = 0
        bm_right_num = 0
        if self.load_from:
            load_model(self.load_from, self)
        for idx, batch in enumerate(tqdm(self.dataloader)):
            answers = defaultdict(list)
            attitude_tensor = []
            attitude_score_tensor = []
            inp = self.preprocess_data(batch)
            gts = batch[self.y]
            self.logger.log(f"\n-------------- batch {idx} ---------------")
            self.logger.log("Ground truth:")
            for i in gts:
                gt = 'positive' if batch[self.y][i] == 1 else 'negative'
                self.logger.log(f"It is {gt}.")
            _, benchmark_ids = self.inference(self.ori_llm, inp)
            bm_ans = self.get_answer_str(self.get_text(benchmark_ids))
            bm_att_tensor = torch.tensor([get_attitude([x]) for x in bm_ans])
            bm_att_sc_tensor = torch.tensor([get_attitude_score([x]) for x in bm_ans])
            bm_right_num += torch.sum(bm_att_tensor == gts)
            self.logger.log("Benchmark answers:")
            for bm in bm_ans:
                self.logger.log(bm)
            for i in range(2):
                self.logger.log(f"Sample answers {i} / 4:")
                _, sample_ids = self.inference(self.llm, inp)
                answer_str = self.get_answer_str(self.get_text(sample_ids))
                for x in answer_str:
                    self.logger.log(x)
                for ii, sample_seq in enumerate(answer_str):
                    answers[ii].append(sample_seq)
            for k,v in answers.items():
                attitude_tensor.append(get_attitude(v))
                attitude_score_tensor.append(get_attitude_score(v))
            attitude_tensor = torch.tensor(attitude_tensor)
            attitude_score_tensor = torch.tensor(attitude_score_tensor)
            right_num += torch.sum(attitude_tensor == gts)
            sample_num += len(batch)
            self.bs_num += 1
            if self.bs_num >= self.iteration:
                break
        self.logger.log("\n\nSummary:")
        self.logger.log(f"Benchmark Accuracy: {bm_right_num} / {sample_num} = {(bm_right_num / sample_num).item()}")
        self.logger.log(f"Accuracy: {right_num} / {sample_num} = {(right_num / sample_num).item()}")


class GrammaticalAcceptableTrainer(Trainer):
    def __init__(self, model, data, conf) -> None:
        super().__init__(model, data, conf)

    def prepare_para(self):
        pass

    def preprocess_data(self, batch, with_answer=False):
        if with_answer:
            for i in range(len(batch[self.x])):
                ans = f" (It is {'acceptable' if batch[self.y][i] == 1 else 'unacceptable'}.)"
                batch[self.x][i] = batch[self.x][i] + ans + " --- Is this sentence grammatically acceptable/unacceptable?"
            batch_texts = batch[self.x]
        else:
            batch_texts = [data + " --- Is this sentence grammatically acceptable/unacceptable?" for data in batch[self.x]]
        inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        inputs.to(self.device)
        return inputs.input_ids

    def inference(self, model, x):
        ans = model.generate(x, max_new_tokens=8, return_dict_in_generate=True, output_scores=True)
        if self.use_score:
            return torch.stack(ans.scores).permute(1,0,2), ans.sequences
        else:
            # try:
            score = self.score_list()
            return score, ans.sequences
            # except:
            #     raise NotImplementedError("There is no score_list function implemented when use score = False.")
            # feature_extractor = pipeline(task="text-generation", tokenizer=self.tokenizer, model=model, device=self.device)
            # generate_kargs = {
            #     "max_new_tokens": 8,
            #     "output_scores": True
            # }
            # feats = feature_extractor(x_text, **generate_kargs)
            # new_token_features = []
            # for i, feat in enumerate(feats):
            #     input_token_count = len(self.tokenizer(x_text[i])["input_ids"])

            #     new_token_feat = torch.tensor(feat)[:, input_token_count:, :]
            #     new_token_features.append(new_token_feat)
        # return self.align_tensors(new_token_features), ans.sequences

    def get_text(self, sequences_ids):
        decoded_sequences = [self.tokenizer.decode(output, skip_special_tokens=False) for output in sequences_ids]
        return decoded_sequences

    def align_tensors(self, tensor_list):
        max_shape = tuple(max(tensor.shape[dim] for tensor in tensor_list if dim < len(tensor.shape))
                        for dim in range(max(len(tensor.shape) for tensor in tensor_list)))
        padded_tensors = []
        for tensor in tensor_list:
            padding_config = []
            for dim in range(len(max_shape)):
                padding_needed = max_shape[dim] - tensor.shape[dim] if dim < len(tensor.shape) else max_shape[dim]
                padding_config.extend([0, padding_needed])
            tensor_padded = F.pad(tensor, padding_config)
            padded_tensors.append(tensor_padded)
        self.logger.log(f"Aligned tensors to shape {max_shape}.")
        return padded_tensors


    def run(self):
        self.set_seed(31415926)
        self.llm.train()
        self.step = 0
        if self.load_from:
            load_model(self.load_from, self)
        for i in range(self.epochs):
            for batch in tqdm(self.dataloader):
                # try:
                inp = self.preprocess_data(batch)
                inp_with_ans = self.preprocess_data(batch, with_answer=True)
                benchmark, benchmark_ids = self.inference(self.ori_llm, inp_with_ans)
                benchmark_seqs = self.get_text(benchmark_ids)

                self.logger.log("\n---------------------------------------------")
                self.logger.log("Benchmark: ")
                for benchmark_seq in benchmark_seqs:
                    self.logger.log(benchmark_seq)
                self.logger.log("--------------------------------------------- \n")

                loss_1 = 0.0
                loss_2 = 0.0

                for j in range(5):
                    sample_feature, one_sample_ids = self.inference(self.llm, inp)
                    if sample_feature.shape != benchmark.shape:
                        [sample_feature, benchmark] = self.align_tensors([sample_feature, benchmark])

                    # Get the sample sequence and print them out
                    samples_seqs = self.get_text(one_sample_ids)
                    self.logger.log(f"Sample {j + 1}: ")
                    for sample_seq in samples_seqs:
                        self.logger.log(sample_seq)

                    # sample_features.append(sample_feature)
                    loss_1 += torch.dist(self.feature_map()['rec'], self.feature_map()['ori'], p=2)
                    loss_2 += torch.dist(sample_feature, benchmark, p=2)
                    # loss_2 += torch.dist(torch.clamp(F.softmax(sample_feature), 0, 1),
                    #                      torch.clamp(F.softmax(benchmark), 0, 1),
                    #                      p=2
                    #                     )
                    # loss_2 += torch.dist(torch.where(sample_feature == float('-inf'),
                    #                                  torch.zeros_like(sample_feature),
                    #                                  sample_feature
                    #                                 ),
                    #                      torch.where(benchmark == float('-inf'),
                    #                                  torch.zeros_like(benchmark),
                    #                                  benchmark
                    #                                 ),
                    #                      p=2
                    #                     )
                    loss = (1 - self.lambda_) * loss_1 + self.lambda_ * loss_2

                loss /= 5
                loss_1 /= 5
                loss_2 /= 5

                self.logger.log(f"ITRATION: {self.step}")
                self.logger.log(f"LAMBDA: {self.lambda_}")
                self.logger.log(f"LOSS: {loss}")
                self.logger.log(f"LOSS 1: {loss_1}")
                self.logger.log(f"LOSS 2: {loss_2}")
                self.writer_.add_scalar("Loss 1", loss_1.float(), self.step)
                self.writer_.add_scalar("Loss 2", loss_2.float(), self.step)
                self.writer_.add_scalar("Loss", loss.float(), self.step)
                self.step += 1
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.lambda_ = self.lambda_scheduler.update(self.step, 4276)

                if self.step % 50 == 0:
                    checkpoint = {'epoch': i + 1, 'step': self.step + 1, 'model_state_dict': self.llm.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'loss': loss}
                    torch.save(checkpoint, os.path.join(self.ckpt_path, f"epoch-{i + 1}_step-{self.step}.pth"))
                clean_and_link_checkpoints(self.ckpt_path)
                get_gpu_memory()

            self.scheduler.step()


# if __name__ == '__main__':
#     from src.model import glm
#     from src.data import twitter_data