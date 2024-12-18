import logging
import pickle

import numpy as np
from sklearn import metrics
from torch.backends import cudnn
import torch
from torch import nn
import warnings
from model.NANS_KoBeM import NANS_KoBeM
from dataloader import NANS_KoBeM_DataLoader
from torch.autograd import Variable

from utils.min_norm_solvers import MinNormSolver


warnings.filterwarnings("ignore")
cudnn.benchmark = True


class trainer(object):
    def __init__(self, config, data, ref_vec, pref_idx):
        super(trainer, self).__init__()
        self.config = config
        self.logger = logging.getLogger("trainer")
        self.metric = config.metric

        self.mode = config.mode
        self.manual_seed = config.seed
        self.device = torch.device("cpu")

        self.current_epoch = 1
        self.current_iteration = 1

        if self.metric == "rmse":
            self.best_val_perf = 1.
        elif self.metric == "auc":
            self.best_val_perf = 0.
        self.best_val_perf_material = 0.
        self.train_loss_list = []
        self.train_loss_material_list = []
        self.train_loss_all_list = []
        self.test_loss_list = []
        self.test_loss_material_list = []
        self.test_loss_all_list = []
        self.test_roc_auc_list = []
        self.test_pr_auc_list = []
        self.test_rmse_list = []
        self.test_hr_material_list = []
        self.test_ndcg_material_list = []
        self.test_mrr_material_list = []

        self.weights = []
        self.task_train_losses = []
        self.task_test_losses = []
        self.train_evals = []
        self.test_evals = []
        self.ref_vec = ref_vec
        self.pref_idx = pref_idx

        self.ref_vec = self.ref_vec.to(self.device)
        # self.pref_idx= self.pref_idx.to(self.device)

        self.data_loader = NANS_KoBeM_DataLoader(config, data)
        self.model = NANS_KoBeM(config)

        # self.criterion = nn.MSELoss(reduction='sum')
        self.criterion = nn.BCELoss(reduction='sum')
        if config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.config.learning_rate,
                                             momentum=self.config.momentum,
                                             weight_decay=self.config.weight_decay)
        elif config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.config.learning_rate,
                                              betas=(config.beta1, config.beta2),
                                              eps=self.config.epsilon,
                                              weight_decay=self.config.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=0,
            min_lr=1e-5,
            factor=0.5,
            verbose=True
        )

        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            # torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.criterion = self.criterion.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print("Program will run on *****GPU-CUDA***** ")
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")
            print("Program will run on *****CPU*****\n")




    def train(self):
        # print the current preference vector
        print('Preference Vector ({}/{}):'.format(self.pref_idx + 1, self.config.num_pref))
        print(self.ref_vec[self.pref_idx].cpu().numpy())

        #find the initial solution, stop early once a feasible solution is found, usually can be found with a few steps
        self.train_find_init_solu(2)

        # run niter epochs of ParetoMTL

        for epoch in range(1, self.config.max_epoch + 1):
            print("=" * 50 + "Epoch {}".format(epoch) + "=" * 50)
            self.train_one_epoch()
            self.validate()
            self.current_epoch += 1



    def train_find_init_solu(self, init_epochs):
        # run at most 2 epochs to find the initial solution
        # stop early once a feasible solution is found
        # usually can be found with a few steps
        for t in range(init_epochs):

            self.model.train()
            for batch_idx, data in enumerate(self.data_loader.train_loader):
                q_list, a_list, target_answers_list, target_masks_list = data

                q_list = q_list.to(self.device)
                a_list = a_list.to(self.device)
                target_answers_list = target_answers_list.to(self.device)
                target_masks_list = target_masks_list.to(self.device)

                neg_q_list = self.generate_negative_q(q_list)

                self.optimizer.zero_grad()
                # output, output_material = self.model(q_list, a_list)
                output, output_material = self.model(q_list, a_list)

                label = torch.masked_select(target_answers_list[:, 1:], target_masks_list[:, 1:])

                output = torch.masked_select(output, target_masks_list[:, 1:])
                loss_q = self.criterion(output.float(), label.float())

                output_material_pos = torch.gather(output_material, 2, (q_list[:, 1:]).unsqueeze(-1)).squeeze(-1)
                output_material_neg = torch.gather(output_material, 2, (neg_q_list[:, 1:]).unsqueeze(-1)).squeeze(-1)

                output_material_pos = torch.masked_select(output_material_pos, target_masks_list[:, 1:])
                output_material_neg = torch.masked_select(output_material_neg, target_masks_list[:, 1:])

                label_material_pos = torch.ones_like(output_material_pos)
                label_material_neg = torch.zeros_like(output_material_neg)

                output_material = torch.cat((output_material_pos, output_material_neg), dim=0)
                label_material = torch.cat((label_material_pos, label_material_neg), dim=0)
                loss_m = self.criterion(output_material.float(), label_material.float())

                task_loss = torch.stack([loss_q, loss_m])

                # obtain and store the gradient value
                grads = {}
                losses_vec = []

                for i in range(self.config.n_tasks):
                    # self.optimizer.zero_grad()
                    # task_loss = model(X, ts)
                    losses_vec.append(task_loss[i].data)

                    task_loss[i].backward(retain_graph=True)

                    grads[i] = []

                    # can use scalable method proposed in the MOO-MTL paper for large scale problem
                    # but we keep use the gradient of all parameters in this experiment
                    for param in self.model.parameters():
                        if param.grad is not None:
                            grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

                grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
                grads = torch.stack(grads_list)

                # calculate the weights
                losses_vec = torch.stack(losses_vec)
                # losses_vec = losses_vec.to(self.device)
                flag, weight_vec = self.get_d_paretomtl_init(grads, losses_vec, self.ref_vec, self.pref_idx)

                # print(flag, weight_vec)

                # early stop once a feasible solution is obtained
                if flag == True:
                    print("fealsible solution is obtained.")
                    break

                # optimization step
                # self.optimizer.zero_grad()
                for i in range(self.config.n_tasks):
                    # task_loss = model(X, ts)
                    self.optimizer.zero_grad()
                    output, output_material = self.model(q_list, a_list)

                    label = torch.masked_select(target_answers_list[:, 1:], target_masks_list[:, 1:])

                    output = torch.masked_select(output, target_masks_list[:, 1:])
                    loss_q = self.criterion(output.float(), label.float())

                    output_material_pos = torch.gather(output_material, 2, (q_list[:, 1:]).unsqueeze(-1)).squeeze(-1)
                    output_material_neg = torch.gather(output_material, 2, (neg_q_list[:, 1:]).unsqueeze(-1)).squeeze(
                        -1)

                    output_material_pos = torch.masked_select(output_material_pos, target_masks_list[:, 1:])
                    output_material_neg = torch.masked_select(output_material_neg, target_masks_list[:, 1:])

                    label_material_pos = torch.ones_like(output_material_pos)
                    label_material_neg = torch.zeros_like(output_material_neg)

                    output_material = torch.cat((output_material_pos, output_material_neg), dim=0)
                    label_material = torch.cat((label_material_pos, label_material_neg), dim=0)
                    loss_m = self.criterion(output_material.float(), label_material.float())

                    task_loss = torch.stack([loss_q, loss_m])

                    if i == 0:
                        loss_total = weight_vec[i] * task_loss[i]
                    else:
                        loss_total = loss_total + weight_vec[i] * task_loss[i]

                loss_total.backward()
                self.optimizer.step()

            else:
                # continue if no feasible solution is found
                continue
            # break the loop once a feasible solutions is found
            break

        # print('')



    def train_one_epoch(self):
        self.model.train()
        self.logger.info("\n")
        self.logger.info("Train Epoch: {}".format(self.current_epoch))
        self.logger.info("learning rate: {}".format(self.optimizer.param_groups[0]['lr']))
        self.train_loss = 0
        self.train_loss_material = 0
        train_elements = 0
        train_elements_material = 0

        for batch_idx, data in enumerate(self.data_loader.train_loader):
            q_list, a_list, target_answers_list, target_masks_list = data

            q_list = q_list.to(self.device)
            a_list = a_list.to(self.device)
            target_answers_list = target_answers_list.to(self.device)
            target_masks_list = target_masks_list.to(self.device)

            neg_q_list = self.generate_negative_q(q_list)

            self.optimizer.zero_grad()
            # output, output_material = self.model(q_list, a_list)
            output, output_material = self.model(q_list, a_list)

            label = torch.masked_select(target_answers_list[:, 1:], target_masks_list[:, 1:])

            output = torch.masked_select(output, target_masks_list[:, 1:])
            loss_q = self.criterion(output.float(), label.float())

            output_material_pos = torch.gather(output_material, 2, (q_list[:, 1:]).unsqueeze(-1)).squeeze(-1)
            output_material_neg = torch.gather(output_material, 2, (neg_q_list[:, 1:]).unsqueeze(-1)).squeeze(-1)

            output_material_pos = torch.masked_select(output_material_pos, target_masks_list[:, 1:])
            output_material_neg = torch.masked_select(output_material_neg, target_masks_list[:, 1:])

            label_material_pos = torch.ones_like(output_material_pos)
            label_material_neg = torch.zeros_like(output_material_neg)

            output_material = torch.cat((output_material_pos, output_material_neg), dim=0)
            label_material = torch.cat((label_material_pos, label_material_neg), dim=0)
            loss_m = self.criterion(output_material.float(), label_material.float())

            task_loss = torch.stack([loss_q, loss_m])

            # obtain and store the gradient
            grads = {}
            losses_vec = []

            for i in range(self.config.n_tasks):
                # self.optimizer.zero_grad()
                # task_loss = model(X, ts)
                losses_vec.append(task_loss[i].data)

                task_loss[i].backward(retain_graph=True)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                grads[i] = []

                # can use scalable method proposed in the MOO-MTL paper for large scale problem
                # but we keep use the gradient of all parameters in this experiment
                for param in self.model.parameters():
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            grads = torch.stack(grads_list)

            # calculate the weights
            losses_vec = torch.stack(losses_vec)
            weight_vec = self.get_d_paretomtl(grads, losses_vec, self.ref_vec, self.pref_idx)

            normalize_coeff = self.config.n_tasks / torch.sum(torch.abs(weight_vec))
            weight_vec = weight_vec * normalize_coeff

            # optimization step
            self.optimizer.zero_grad()
            for i in range(len(task_loss)):
                # task_loss = model(X, ts)
                output, output_material = self.model(q_list, a_list)

                label = torch.masked_select(target_answers_list[:, 1:], target_masks_list[:, 1:])

                output = torch.masked_select(output, target_masks_list[:, 1:])
                loss_q = self.criterion(output.float(), label.float())

                output_material_pos = torch.gather(output_material, 2, (q_list[:, 1:]).unsqueeze(-1)).squeeze(-1)
                output_material_neg = torch.gather(output_material, 2, (neg_q_list[:, 1:]).unsqueeze(-1)).squeeze(-1)

                output_material_pos = torch.masked_select(output_material_pos, target_masks_list[:, 1:])
                output_material_neg = torch.masked_select(output_material_neg, target_masks_list[:, 1:])

                label_material_pos = torch.ones_like(output_material_pos)
                label_material_neg = torch.zeros_like(output_material_neg)

                output_material = torch.cat((output_material_pos, output_material_neg), dim=0)
                label_material = torch.cat((label_material_pos, label_material_neg), dim=0)
                loss_m = self.criterion(output_material.float(), label_material.float())

                task_loss = torch.stack([loss_q, loss_m])
                if i == 0:
                    loss_total = weight_vec[i] * task_loss[i]
                else:
                    loss_total = loss_total + weight_vec[i] * task_loss[i]


            self.train_loss += loss_q.item()
            train_elements += target_masks_list[:, 1:].int().sum()
            self.train_loss_material += loss_m.item()
            train_elements_material += (target_masks_list[:, 1:].int().sum()) * 2

            self.weight_vec = weight_vec
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()


        self.train_loss = self.train_loss / train_elements
        self.train_loss_material = self.train_loss_material / train_elements_material
        self.train_loss_all = self.train_loss + self.config.weight_material * self.train_loss_material
        self.scheduler.step(self.train_loss + self.config.weight_material * self.train_loss_material)
        # self.scheduler.step(self.train_loss)
        self.train_loss_list.append(self.train_loss.data.cpu())
        self.train_loss_material_list.append((self.config.weight_material * self.train_loss_material).data.cpu())
        self.train_loss_all_list.append(self.train_loss_all.data.cpu())
        # self.logger.info("Train Loss: {:.6f}".format(self.train_loss))
        self.logger.info(
            "Train Loss: {:.6f}, Train Loss Material: {:.6f}".format(self.train_loss, self.train_loss_material))
        # print("Train Loss: {:.6f}".format(self.train_loss))
        print("Train Loss: {:.6f}, Train Loss Material: {:.6f}".format(self.train_loss, self.train_loss_material))


    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        if self.mode == "train":
            self.logger.info("Validation Result at Epoch: {}".format(self.current_epoch))
            # print("Validation Result at Epoch: {}".format(self.current_epoch))
        else:
            self.logger.info("Test Result at Epoch: {}".format(self.current_epoch))
            # print("Test Result at Epoch: {}".format(self.current_epoch))

        self.test_loss = 0
        self.test_loss_material = 0
        pred_labels = []
        true_labels = []
        pred_labels_material = []
        true_labels_material = []
        with torch.no_grad():
            for data in self.data_loader.test_loader:
                q_list, a_list, target_answers_list, target_masks_list, neg_evaluation_list = data
                q_list = q_list.to(self.device)
                a_list = a_list.to(self.device)
                target_answers_list = target_answers_list.to(self.device)
                target_masks_list = target_masks_list.to(self.device)
                neg_evaluation_list = neg_evaluation_list.to(self.device)

                neg_q_list = self.generate_negative_q(q_list)

                # output, output_material = self.model(q_list, a_list, l_list, d_list)
                output, output_material = self.model(q_list, a_list)

                # self.test_output_save = output.detach().clone()
                # self.test_label_save = target_answers_list.detach().clone()
                # self.test_mask_save = target_masks_list.detach().clone()

                label = torch.masked_select(target_answers_list[:, 1:], target_masks_list[:, 1:])

                output = torch.masked_select(output, target_masks_list[:, 1:])
                test_loss_q = self.criterion(output.float(), label.float()).item()


                output_material_pos = torch.gather(output_material, 2, (q_list[:, 1:]).unsqueeze(-1)).squeeze(-1)
                output_material_neg = torch.gather(output_material, 2, (neg_q_list[:, 1:]).unsqueeze(-1)).squeeze(-1)

                output_material_pos = torch.masked_select(output_material_pos, target_masks_list[:, 1:])
                output_material_neg = torch.masked_select(output_material_neg, target_masks_list[:, 1:])

                label_material_pos = torch.ones_like(output_material_pos)
                label_material_neg = torch.zeros_like(output_material_neg)

                output_material_all = torch.cat((output_material_pos, output_material_neg), dim=0)
                label_material_all = torch.cat((label_material_pos, label_material_neg), dim=0)
                test_loss_m = self.criterion(output_material_all.float(), label_material_all.float())

                # test_loss = test_loss_q + self.config.weight_material * test_loss_m

                pred_labels.extend(output.tolist())
                true_labels.extend(label.tolist())
                # print(list(zip(true_labels, pred_labels)))

                material_id = torch.cat((q_list.unsqueeze(-1), neg_evaluation_list), dim = 2)
                output_material_eva = torch.gather(output_material, 2, material_id[:, 1:, :])

                # material_id = material_id[:, 1:, :][target_masks_list[:,1:]]
                output_material_eva = output_material_eva[target_masks_list[:,1:]]

                test_elements = target_masks_list[:, 1:].int().sum()
                test_elements_material = (target_masks_list[:, 1:].int().sum())*2

                self.test_loss = test_loss_q / test_elements
                self.test_loss_material = test_loss_m / test_elements_material
                self.test_loss_all = self.test_loss + self.config.weight_material * self.test_loss_material
                self.test_loss_list.append(self.test_loss.data.cpu())
                self.test_loss_material_list.append(
                    (self.config.weight_material * self.test_loss_material).data.cpu())
                self.test_loss_all_list.append(self.test_loss_all.data.cpu())

                print("Test Loss: {:.6f}, Test Loss Material: {:.6f}".format(self.test_loss, self.test_loss_material))
        self.track_best(true_labels, pred_labels)
        self.track_best_material(output_material_eva)


    def track_best(self, true_labels, pred_labels):
        self.pred_labels = np.array(pred_labels).squeeze()
        self.true_labels = np.array(true_labels).squeeze()
        self.logger.info(
            "pred size: {} true size {}".format(self.pred_labels.shape, self.true_labels.shape))
        if self.metric == "rmse":
            perf = np.sqrt(metrics.mean_squared_error(self.true_labels, self.pred_labels))
            perf_mae = metrics.mean_absolute_error(self.true_labels, self.pred_labels)
            self.logger.info('RMSE: {:.05}'.format(perf))
            print('RMSE: {:.05}'.format(perf))
            self.logger.info('MAE: {:.05}'.format(perf_mae))
            print('MAE: {:.05}'.format(perf_mae))
            if perf < self.best_val_perf:
                self.best_val_perf = perf
                self.best_train_loss = self.train_loss.item()
                self.best_test_loss = self.test_loss.item()
                self.best_epoch = self.current_epoch


            self.test_roc_auc_list.append(perf)
            self.test_pr_auc_list.append(perf_mae)
        elif self.metric == "auc":
            perf = metrics.roc_auc_score(self.true_labels, self.pred_labels)
            prec, rec, _ = metrics.precision_recall_curve(self.true_labels, self.pred_labels)
            pr_auc = metrics.auc(rec, prec)
            self.logger.info('ROC-AUC: {:.05}'.format(perf))
            print('ROC-AUC: {:.05}'.format(perf))
            self.logger.info('PR-AUC: {:.05}'.format(pr_auc))
            print('PR-AUC: {:.05}'.format(pr_auc))
            if perf > self.best_val_perf:
                self.best_val_perf = perf
                self.best_train_loss = self.train_loss.item()
                self.best_test_loss = self.test_loss.item()
                self.best_epoch = self.current_epoch

            self.test_roc_auc_list.append(perf)
            self.test_pr_auc_list.append(pr_auc)
        else:
            raise AttributeError



    def track_best_material(self, output_material_eva):

        hr = []
        ndcg = []
        mrr = []
        for i in output_material_eva:
            i = np.array(i)
            sort = np.argsort(i)[::-1][:self.config.top_k_eva]
            hr_arr = 0
            ndcg_arr = 0
            mrr_arr = 0
            if 0 in sort:
                pos = np.where(sort == 0)[0][0]
                hr_arr = 1.0
                ndcg_arr = np.log(2) / np.log(pos + 2.0)
                mrr_arr = 1.0 / (pos + 1.0)
            hr.append(hr_arr)
            ndcg.append(ndcg_arr)
            mrr.append(mrr_arr)

        hr = np.mean(hr)
        ndcg = np.mean(ndcg)
        mrr = np.mean(mrr)

        self.logger.info('HR: {:.05}'.format(hr))
        print('HR: {:.05}'.format(hr))
        self.logger.info('NDCG: {:.05}'.format(ndcg))
        print('NDCG: {:.05}'.format(ndcg))
        self.logger.info('MRR: {:.05}'.format(mrr))
        print('MRR: {:.05}'.format(mrr))

        if hr < self.best_val_perf_material:
            self.best_val_perf_material = hr
            self.best_train_loss_material = self.train_loss_material.item()
            self.best_test_loss_material = self.test_loss_material.item()
            self.best_epoch_material = self.current_epoch


        self.test_hr_material_list.append(hr)
        self.test_ndcg_material_list.append(ndcg)
        self.test_mrr_material_list.append(mrr)

        # return hr_arr, ndcg_arr, mrr_arr

    def generate_negative_q(self, q_list):
        neg_q_list = []
        # quesiton_list = np.arange(1, self.data_loader.num_questions+1, dtype=int)
        for seq in q_list:
            neg_items_seq = []
            for q_item in seq:
                # q_neighbors_weights = self.data_loader.q_q_neighbors_weights[q_item, 1:]#1: because dont want to sample the padding item
                # if np.sum(q_neighbors_weights) == 0:
                #     neg_q_item = np.random.choice(quesiton_list) #random sample for item without neighbor
                # else:
                #
                #     neg_q_item = np.random.choice(quesiton_list, p=q_neighbors_weights)

                neg_q_item = self.data_loader.q_q_neighbors[q_item][torch.randint(self.config.k_neighbors, (1,))]
                neg_items_seq.append(neg_q_item)
            neg_q_list.append(neg_items_seq)

        return torch.Tensor(neg_q_list).long()



    def get_d_paretomtl_init(self, grads, value, ref_vecs, i):
        """
        calculate the gradient direction for ParetoMTL initialization
        """

        flag = False
        nobj = value.shape

        # check active constraints
        current_weight = ref_vecs[i]
        rest_weights = ref_vecs
        w = rest_weights - current_weight

        w = w.to(self.device)
        value = value.to(self.device)

        gx = torch.matmul(w, value / torch.norm(value))
        idx = gx > 0

        # calculate the descent direction
        if torch.sum(idx) <= 0:
            flag = True
            return flag, torch.zeros(nobj)
        if torch.sum(idx) == 1:
            # sol = torch.ones(1).cuda().float()
            sol = torch.ones(1).float()
        else:
            vec = torch.matmul(w[idx], grads)
            sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])

        weight0 = torch.sum(torch.stack([sol[j] * w[idx][j, 0] for j in torch.arange(0, torch.sum(idx))]))
        weight1 = torch.sum(torch.stack([sol[j] * w[idx][j, 1] for j in torch.arange(0, torch.sum(idx))]))
        weight = torch.stack([weight0, weight1])

        return flag, weight

    def get_d_paretomtl(self, grads, value, ref_vecs, i):
        """ calculate the gradient direction for ParetoMTL """

        # check active constraints
        current_weight = ref_vecs[i]
        rest_weights = ref_vecs
        w = rest_weights - current_weight

        w = w.to(self.device)
        value = value.to(self.device)

        gx = torch.matmul(w, value / torch.norm(value))
        idx = gx > 0

        # calculate the descent direction
        if torch.sum(idx) <= 0:
            sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
            # return torch.tensor(sol).cuda().float()
            return torch.tensor(sol).float()

        vec = torch.cat((grads, torch.matmul(w[idx], grads)))
        sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])

        weight0 = sol[0] + torch.sum(
            torch.stack([sol[j] * w[idx][j - 2, 0] for j in torch.arange(2, 2 + torch.sum(idx))]))
        weight1 = sol[1] + torch.sum(
            torch.stack([sol[j] * w[idx][j - 2, 1] for j in torch.arange(2, 2 + torch.sum(idx))]))
        weight = torch.stack([weight0, weight1])

        return weight



