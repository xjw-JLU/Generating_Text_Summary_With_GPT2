import argparse
from datetime import datetime
import os
import time

import numpy as np
from transformers import GPT2LMHeadModel,AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from tqdm import tnrange, tqdm, trange

from dataset import GPT21024Dataset 
from utils import add_special_tokens, generate_sample, set_seed
import json

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from rouge import Rouge



# def train(args, model, tokenizer, train_dataset, valid_dataset, ignore_index):
#     """ Trains GPT2 model and logs necessary details.
#         Args:
#             args: dict that contains all the necessary information passed by user while training
#             model: finetuned gpt/gpt2 model
#             tokenizer: GPT/GPT2 tokenizer
#             train_dataset: GPT21024Dataset object for training data
#             ignore_index: token not considered in loss calculation
#     """
#     writer = SummaryWriter('./logs')
#     # train_sampler = RandomSampler(train_dataset)


#     train_sampler = DistributedSampler(train_dataset)
    
    
#     train_dl = DataLoader(train_dataset,sampler=train_sampler,batch_size=args.batch_size,num_workers=args.num_workers)
#     loss_fct = CrossEntropyLoss(ignore_index=ignore_index) #ignores padding token for loss calculation
#     optimizer = AdamW(model.parameters(),lr=args.lr)
#     scheduler = get_linear_schedule_with_warmup(optimizer,100,80000)

#     #-----
#     training_results = {}
#     #-----

#     global_step = 0
#     tr_loss, logging_loss = 0.0, 0.0
#     model.zero_grad()
#     train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
#     set_seed(args)
#     for  epoch, _ in enumerate(train_iterator):

#         #-----
#         train_sampler.set_epoch(epoch)
#         #-----

#         epoch_iterator = tqdm(train_dl, desc="Training")
#         for step, batch in enumerate(epoch_iterator):
#             # inputs, labels = torch.tensor(batch['article']), torch.tensor(batch['article'])
#             # inputs, labels = batch['article'].clone().detach(), batch['article'].clone().detach()
#             inputs, labels = batch['article'].clone().detach().to(args.device), batch['article'].clone().detach().to(args.device)
            
#             #-----
#             sample_ids = batch['id']

#             # inputs = inputs.cuda(non_blocking=True)
#             # labels = labels.cuda(non_blocking=True)
#             #-----

#             # inputs = inputs.to(args.device)
#             # labels = labels.to(args.device)

#             model.train()
#             logits = model(inputs)[0]
#             # idx = batch['sum_idx'].item() # index of separator token

#             idx = batch['sum_idx'].to(args.device)  # 应该是一个向量
#             batch_loss = 0
#             for i in range(args.batch_size):
#             # only consider loss on reference summary just like seq2seq models
#             # shift_logits = logits[..., idx:-1, :].contiguous()
#             # shift_labels = labels[..., idx+1:].contiguous()
#             # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#             # loss = loss/args.gradient_accumulation_steps
#                 current_idx = idx[i].item()
#                 shift_logits = logits[i, current_idx:-1, :].unsqueeze(0).contiguous()
#                 shift_labels = labels[i, current_idx+1:].unsqueeze(0).contiguous()
#                 loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#                 loss = loss / args.gradient_accumulation_steps
#                 batch_loss += loss
#             #-----
#             # sample_losses = loss.view(-1, shift_logits.size(-2)).mean(dim=1)
#             sample_perplexities = torch.exp(torch.tensor(loss.item()))
#             # sample_perplexities = torch.exp(sample_losses)
#             # for i, sid in enumerate(sample_ids):
#             #     training_results[sid] = {
#             #         'loss': loss[i].item(),  # 获取每个样本的loss并将其转换为Python数字
#             #         'perplexity': sample_perplexities[i].item()  # 获取每个样本的perplexity并将其转换为Python数字
#             #     }
#             for i, sid in enumerate(sample_ids):
#                 training_results[sid.item()] = {
#                     'loss': loss.item(),  # 获取每个样本的loss并将其转换为Python数字
#                     'perplexity': sample_perplexities.item()  # 获取每个样本的perplexity并将其转换为Python数字
#                 }

#             # for i, sid in enumerate(sample_ids):
#             #     training_results[str(sid)] = {
#             #         'loss': str(loss),  # 获取每个样本的loss并将其转换为Python数字
#             #         'perplexity': str(sample_perplexities)  # 获取每个样本的perplexity并将其转换为Python数字
#             #     }

#             output_training_results_file = os.path.join(args.output_dir, "training_results.json")
#             with open(output_training_results_file, "w") as f:  
#                 json.dump(training_results, f, indent=4)    
#             #-----

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
#             tr_loss += loss.item()
#             if (step + 1) % args.gradient_accumulation_steps == 0:
#                 optimizer.step()
#                 scheduler.step()  # Update learning rate schedule
#                 model.zero_grad()
#                 global_step += 1
#                 writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
#                 writer.add_scalar('loss', (tr_loss - logging_loss)/args.gradient_accumulation_steps, global_step)
#                 logging_loss = tr_loss
#                 print("loss:", loss.item(), end='\n\n')
#                 if (step + 1)/args.gradient_accumulation_steps == 1.0:
#                     print('After 1st update: ', end='\n\n')
#                     # generate_sample(valid_dataset, tokenizer, model, num=2, eval_step=False, device=args.device)
                
                
#             if (step + 1) % (10*args.gradient_accumulation_steps) == 0:
#                 results = evaluate(args, model, valid_dataset, ignore_index, global_step)
#                 for key, value in results.items():
#                     writer.add_scalar('eval_{}'.format(key), value, global_step)
#                 print('After', global_step+1,'updates: ', end='\n\n')
#                 # generate_sample(valid_dataset, tokenizer, model, num=2, eval_step=True, device=args.device)
    
#         #-----
#         output_training_results_file = os.path.join(args.output_dir, "training_results.json")
#         with open(output_training_results_file, "w") as f:  
#             json.dump(training_results, f, indent=4)                
        
#         #-----


def train(args, model, tokenizer, train_dataset, valid_dataset, ignore_index):
    """ Trains GPT2 model and logs necessary details.
        Args:
            args: dict that contains all the necessary information passed by user while training
            model: finetuned gpt/gpt2 model
            tokenizer: GPT/GPT2 tokenizer
            train_dataset: GPT21024Dataset object for training data
            ignore_index: token not considered in loss calculation
    """
    writer = SummaryWriter('./logs')
    # train_sampler = RandomSampler(train_dataset)


    train_sampler = DistributedSampler(train_dataset)
    
    
    train_dl = DataLoader(train_dataset,sampler=train_sampler,batch_size=args.batch_size,num_workers=args.num_workers)
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index) #ignores padding token for loss calculation
    optimizer = AdamW(model.parameters(),lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer,100,80000)

    #-----

    #-----

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    for  epoch, _ in enumerate(train_iterator):
        training_results = {}

        test_results = {}
        #-----
        train_sampler.set_epoch(epoch)
        #-----

        train_samples = []

        epoch_iterator = tqdm(train_dl, desc="Training")
        for step, batch in enumerate(epoch_iterator):
            # inputs, labels = torch.tensor(batch['article']), torch.tensor(batch['article'])
            # inputs, labels = batch['article'].clone().detach(), batch['article'].clone().detach()
            inputs, labels = batch['article'].clone().detach().to(args.device), batch['article'].clone().detach().to(args.device)
            
            #-----
            sample_ids = batch['id']
            train_samples = train_samples + sample_ids.tolist()

            model.train()
            logits = model(inputs)[0]
            # idx = batch['sum_idx'].item() # index of separator token
            idx = batch['sum_idx'].to(args.device)
            # only consider loss on reference summary just like seq2seq models
            batch_loss = 0
            for i, sid in enumerate(sample_ids): # 遍历批次中的每个样本
                current_idx = idx[i].item()
                shift_logits = logits[i, current_idx:-1, :].unsqueeze(0).contiguous()
                shift_labels = labels[i, current_idx+1:].unsqueeze(0).contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                batch_loss += loss  # 累加每个样本的损失
                sample_perplexities = torch.exp(torch.tensor(loss.item()))
                training_results[sid.item()] = {
                    'loss': loss.item(),  # 获取每个样本的loss并将其转换为Python数字
                    'perplexity': sample_perplexities.item()  # 获取每个样本的perplexity并将其转换为Python数字
                }
            
            output_training_results_file = os.path.join(args.output_dir, f"training_results_epoch_{epoch}.json")
            with open(output_training_results_file, "a") as f:  
                json.dump(training_results, f, indent=4)  
            # loss.backward()
            batch_loss = batch_loss / inputs.size(0)  # 计算批次的平均损失
            if args.gradient_accumulation_steps > 1:
                batch_loss = batch_loss / args.gradient_accumulation_steps
            batch_loss.backward()  # 对平均损失执行反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                writer.add_scalar('loss', (tr_loss - logging_loss)/args.gradient_accumulation_steps, global_step)
                logging_loss = tr_loss
                print("loss:", loss.item(), end='\n\n')
                if (step + 1)/args.gradient_accumulation_steps == 1.0:
                    print('After 1st update: ', end='\n\n')
                    # generate_sample(valid_dataset, tokenizer, model, num=2, eval_step=False, device=args.device)
                
                
            # if (step + 1) % (10*args.gradient_accumulation_steps) == 0:
            if (step + 1) % (args.gradient_accumulation_steps) == 0:
                results, sample_results = evaluate(args, model, valid_dataset, ignore_index, global_step)
                test_results[str(train_samples)] = sample_results

                for key, value in results.items():
                    writer.add_scalar('eval_{}'.format(key), value, global_step)
                print('After', global_step+1,'updates: ', end='\n\n')
                # generate_sample(valid_dataset, tokenizer, model, num=2, eval_step=True, device=args.device)
                output_testing_results_file = os.path.join(args.output_dir, f"testing_results_epoch_{epoch}.json")
                with open(output_testing_results_file, "a") as f:  
                    json.dump(test_results, f, indent=4)
        
               

            

        #-----
            # output_training_results_file = os.path.join(args.output_dir, "training_results.json")
            # with open(output_training_results_file, "w") as f:  
            #     json.dump(training_results, f, indent=4)                
            
            
        #-----


def evaluate(args, model, eval_dataset, ignore_index, global_step=None):
    """ Returns perplexity score on validation dataset.
        Args:
            args: dict that contains all the necessary information passed by user while training
            model: finetuned gpt/gpt2 model
            eval_dataset: GPT21024Dataset object for validation data
            global_step: no. of times gradients have backpropagated
            ignore_index: token not considered in loss calculation
    """
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    eval_output_dir = args.output_dir

    results = {}
    # eval_sampler = SequentialSampler(eval_dataset)
    
    #-----
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    #-----


    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=args.num_workers)
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index) #ignores padding token for loss calculation

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    
    #----
    sample_results = {}
    #----
    # training_results = {}
    rouge = Rouge()
    # 设置默认 ROUGE 分数
    default_rouge_score = {
        'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
        'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
        'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
    }
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = torch.tensor(batch['article']).to(args.device), torch.tensor(batch['article']).to(args.device)
        
        #----
        sample_ids = batch['id']
        #----
        
        with torch.no_grad():
            logits = model(inputs)[0]

            idx = batch['sum_idx'].to(args.device)
            # only consider loss on reference summary just like seq2seq models
            batch_loss = 0
            for i, sid in enumerate(sample_ids): # 遍历批次中的每个样本
                current_idx = idx[i].item()
                shift_logits = logits[i, current_idx:-1, :].unsqueeze(0).contiguous()
                shift_labels = labels[i, current_idx+1:].unsqueeze(0).contiguous()
                lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                eval_loss += lm_loss.mean().item()
                tokenizer = add_special_tokens()
                #-----
                generated_ids = torch.argmax(shift_logits.view(-1, shift_logits.size(-1)), dim=-1)
                reference_ids = shift_labels.view(-1)
                # print("xjw")
                # print(f"Shape of generated_ids: {generated_ids.shape}")
                # print(f"Shape of reference_ids: {reference_ids.shape}")
                generated_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
                reference_text = tokenizer.decode(reference_ids.tolist(), skip_special_tokens=True)
                # print("xjw2")
                # print(generated_text)
                # print("xjw3")
                # print(reference_text)                
                # 然后使用转换后的文本计算 ROUGE 分数

                # 检查 generated_text 和 reference_text 是否为空
                if not generated_text.strip() or not any(char.isalpha() for char in generated_text) or not reference_text.strip():
                # if not generated_text.strip() or not reference_text.strip():

                    # print("Either the hypothesis or the reference is empty. Using default ROUGE scores.")
                    rouge_score = default_rouge_score
                else:
                    # 计算 ROUGE 分数
                    
                    try:
                        # 计算 ROUGE 分数
                        rouge_score = rouge.get_scores(generated_text, reference_text)
                    except ValueError as e:
                        print(f"Error calculating ROUGE: {e}")
                        print(f"Generated text (error): {generated_text}")
                        print(f"Reference text (error): {reference_text}")
                        rouge_score = default_rouge_score

                
                # print("xjw4")
                # print(rouge_score)
                batch_loss += lm_loss  # 累加每个样本的损失
                sample_perplexities = torch.exp(torch.tensor(lm_loss.item()))
                sample_results[sid.item()] = {
                    'loss': lm_loss.item(),  # 获取每个样本的loss并将其转换为Python数字
                    'perplexity': sample_perplexities.item(),  # 获取每个样本的perplexity并将其转换为Python数字
                    'rouge': rouge_score
                }

        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }
    print("perplexity:", perplexity.item())

    if global_step:
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as f:
            for key in sorted(result.keys()):
                f.write('\n\n')
                f.write("time = %s, %s = %s, step = %s\n" % (datetime.now().strftime("%d/%m/%Y %H:%M:%S"), key, str(result[key]), str(global_step)))
    

    return result, sample_results           


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default=5e-5, type=float, required=False, help="learning rate")
    parser.add_argument("--seed",default=42, type=int, required=False, help="seed to replicate results")
    parser.add_argument("--n_gpu",default=8, type=int, required=False, help="no of gpu available")
    parser.add_argument("--gradient_accumulation_steps",default=32, type=int, required=False, help="gradient_accumulation_steps")
    parser.add_argument("--batch_size",default=4, type=int, required=False, help="batch_size")
    parser.add_argument("--num_workers",default=4, type=int, required=False, help="num of cpus available")
    parser.add_argument("--device",default=torch.device('cuda'), required=False, help="torch.device object")
    parser.add_argument("--num_train_epochs",default=4, type=int, required=False, help="no of epochs of training")
    parser.add_argument("--output_dir",default='./output', type=str, required=False, help="path to save evaluation results")
    parser.add_argument("--model_dir",default='./weights', type=str, required=False, help="path to save trained model")
    parser.add_argument("--fp16",default=True, type=bool, required=False, help="whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level",default='O0', type=str, required=False, help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
    parser.add_argument("--max_grad_norm",default=1.0, type=float, help="max gradient norm.")
    parser.add_argument("--root_dir",default='./CNN/gpt2_data_all', type=str, help="location of json dataset.")
    parser.add_argument("--ids_file",default='./CNN/ids.json', type=str, help="location of train, valid and test file indexes")
    
    #----
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    #----
    args = parser.parse_args()


    #----

    # 初始化分布式环境
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    #----
    train_data = GPT21024Dataset(args.root_dir,args.ids_file,mode='train',length=192000) #training on only 3000 datasets
    valid_data = GPT21024Dataset(args.root_dir,args.ids_file,mode='valid',length=8000)  #validation on only 500 datasets
    print(len(train_data))
    print(len(valid_data))
    tokenizer = add_special_tokens()
    ignore_idx = tokenizer.pad_token_id
    model = GPT2LMHeadModel.from_pretrained('/ssd3/xiaojingwu/gpt2')
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    #----
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    #----


    start = time.time()
    train(args, model, tokenizer, train_data, valid_data, ignore_idx)
    print('total time: ', (time.time()-start)/60, " minutes", end='\n\n')

    print('Saving trained model...')

    # model_file = os.path.join(args['model_dir'], 'model_{}_data{}_trained_after_{}_epochs_only_sum_loss_ignr_pad.bin'.format(args['fp16_opt_level'],3000,args['num_train_epochs']))
    # config_file = os.path.join(args['model_dir'], 'config_{}_data{}_trained_after_{}_epochs_only_sum_loss_ignr_pad.json'.format(args['fp16_opt_level'],3000,args['num_train_epochs']))
    # torch.save(model.state_dict(), model_file)
    # model.config.to_json_file(config_file)

    #----
    if dist.get_rank() == 0:
        model_file = os.path.join(args['model_dir'], 'model_{}_data{}_trained_after_{}_epochs_only_sum_loss_ignr_pad.bin'.format(args['fp16_opt_level'],3000,args['num_train_epochs']))
        config_file = os.path.join(args['model_dir'], 'config_{}_data{}_trained_after_{}_epochs_only_sum_loss_ignr_pad.json'.format(args['fp16_opt_level'],3000,args['num_train_epochs']))
        torch.save(model.state_dict(), model_file)
        model.config.to_json_file(config_file)
    
    dist.destroy_process_group()
    #----

if __name__ == '__main__':
	main()