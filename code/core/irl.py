from my_utils import *
from core_nn.nn_irl import *
# from core_nn.nn_old import *
import h5py
import torch
""" MaxEnt-IRL. I.e., Adversarial IL with linear loss function. """
class IRL():
    def __init__(self, state_dim, action_dim, args, initialize_net=True, rebuttal=False):
        self.mini_batch_size = args.mini_batch_size
        self.gp_lambda = args.gp_lambda
        self.gp_alpha = args.gp_alpha
        self.gp_center = args.gp_center
        self.gp_lp = args.gp_lp
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = args.gamma

        self.traj_num = 0

        self.rebuttal = rebuttal
        self.load_demo_list(args, verbose=initialize_net)

        if initialize_net:
            self.initilize_nets(args)

    def initilize_nets(self, args):
        self.discrim_net = Discriminator(self.state_dim, self.action_dim, hidden_size=args.hidden_size,
                                         activation=args.activation, clip=args.clip_discriminator).to(device)
        self.optimizer_discrim = torch.optim.Adam(self.discrim_net.parameters(), lr=args.learning_rate_d)

    def load_demo_list(self, args, verbose=True):

        index_worker = []
        self.index_worker_idx = []
        self.m_return_list = []
        index_start = 0
        expert_state_list, expert_action_list, expert_reward_list, expert_mask_list, worker_id_list = [], [], [], [], []

        if args.c_data == 1:
            k_spec_list = args.noise_level_list
            traj_path = "../imitation_data/TRAJ_h5_D1/%s" % (args.env_name)

            """ trajectory description in result files """
            self.traj_name = "traj_type%d_N%d" % (args.c_data, args.demo_file_size)

            if args.noise_type != "normal":
                self.traj_name += ("_%s" % args.noise_type)
            if args.c_data == 1:
                self.traj_name += ("_%0.2f" % k_spec_list[0])

        else:
            k_spec_list = args.noise_level_list
            traj_path = "../imitation_data/TRAJ_h5_D2/%s" % (args.env_name)

            """ trajectory description in result files """
            self.traj_name = "traj_type%d_N%d" % (args.c_data, args.demo_file_size)
            if args.demo_split_k:
                self.traj_name = "traj_type%dK_N%d" % (args.c_data, args.demo_file_size)
            if args.c_data == 1:
                self.traj_name += ("_%0.2f" % k_spec_list[0])


        worker_id = 0
        length = 1000

        for k in range(0, len(k_spec_list)):
            k_spec = k_spec_list[k]

            if args.c_data == 1:
                traj_filename = traj_path + (
                        "/%s_TRAJ-N%d_t%d" % (args.env_name, args.demo_file_size, k_spec))
            else:
                traj_filename = traj_path + (
                            "/%s_TRAJ-N%d_%s%0.2f" % (args.env_name, args.demo_file_size, args.noise_type, k_spec))

            if args.traj_deterministic:
                traj_filename += "_det"

            hf = h5py.File(traj_filename + ".h5", 'r')
            expert_mask = hf.get('expert_masks')[:length]
            expert_mask_list += [expert_mask][:length]
            expert_state_list += [hf.get('expert_states')[:length]]
            expert_action_list += [hf.get('expert_actions')[:length]]
            expert_reward_list += [hf.get('expert_rewards')[:length]]
            reward_array = hf.get('expert_rewards')[:length]
            step_num = expert_mask.shape[0]

            ## Set k=n and K=N. Work and pendulum and lunarlander. The results are not included in the paper.
            if not args.demo_split_k:
                worker_id = k
                worker_id_list += [np.ones(expert_mask.shape) * worker_id]
                self.index_worker_idx += [index_start + np.arange(0, step_num)]
                index_start += step_num

            else:
                ## need to loop through demo until mask = 0, then increase the k counter.
                ## find index in expert_mask where value is 0
                zero_mask_idx = np.where(expert_mask == 0)[0]
                prev_idx = -1
                for i in range(0, len(zero_mask_idx)):
                    worker_id_list += [np.ones(zero_mask_idx[i] - prev_idx) * worker_id]
                    self.index_worker_idx += [index_start + np.arange(0, zero_mask_idx[i] - prev_idx)]
                    index_start += zero_mask_idx[i] - prev_idx

                    worker_id = worker_id + 1
                    prev_idx = zero_mask_idx[i]

            traj_num = step_num - np.sum(expert_mask)
            m_return = np.sum(reward_array) / traj_num

            self.m_return_list += [m_return]

            if verbose:
                print("TRAJ is loaded from %s with traj_num %s, data_size %s steps, and average return %s" % \
                      (colored(traj_filename, p_color), colored(traj_num, p_color),
                       colored(expert_mask.shape[0], p_color), \
                       colored("%.2f" % (m_return), p_color)))

        expert_states = np.concatenate(expert_state_list, axis=0)
        expert_actions = np.concatenate(expert_action_list, axis=0)
        expert_rewards = np.concatenate(expert_reward_list, axis=0)
        expert_masks = np.concatenate(expert_mask_list, axis=0)
        expert_ids = np.concatenate(worker_id_list, axis=0)

        self.real_state_tensor = torch.FloatTensor(expert_states).to(device_cpu)
        self.real_action_tensor = torch.FloatTensor(expert_actions).to(device_cpu)
        self.real_mask_tensor = torch.FloatTensor(expert_masks).to(device_cpu)
        self.real_worker_tensor = torch.LongTensor(expert_ids).to(device_cpu)
        self.data_size = self.real_state_tensor.size(0)
        # self.worker_num = worker_id + 1 # worker_id start at 0?
        self.worker_num = torch.unique(self.real_worker_tensor).size(0)  # much cleaner code
        self.traj_num = self.real_state_tensor.shape[0]
        if verbose:
            print("Total data pairs: %s, K %s, state dim %s, action dim %s, a min %s, a_max %s" % \
                  (colored(self.real_state_tensor.size(0), p_color), colored(self.worker_num, p_color), \
                   colored(self.real_state_tensor.size(1), p_color), colored(self.real_action_tensor.size(1), p_color),
                   colored(torch.min(self.real_action_tensor).numpy(), p_color),
                   colored(torch.max(self.real_action_tensor).numpy(), p_color) \
                   ))

    def save_model(self, path):
        torch.save(self.discrim_net.state_dict(), path)

    def load_model(self, path):
        self.discrim_net.load_state_dict(torch.load(path, map_location='cpu'))

    def compute_reward(self, states, actions, next_states=None, masks=None):
        return self.discrim_net.get_reward(states, actions)

    def index_sampler(self, offset=0):
        return torch.randperm(self.data_size - offset)[0:self.mini_batch_size].to(device_cpu)

    def update_discriminator(self, batch, index, total_step=0):
        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(s_real, a_real)
        x_fake = self.discrim_net.get_reward(s_fake, a_fake)

        loss_real = x_real.mean()
        loss_fake = x_fake.mean()
        loss = -(loss_real - loss_fake)

        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()

        w_dist = (loss_real - loss_fake).cpu().detach().numpy()
        return w_dist, loss.cpu().detach().numpy()

    def gp_regularizer(self, sa_real, sa_fake):
        if self.gp_lambda == 0:
            return 0

        real_data = sa_real.data
        fake_data = sa_fake.data

        if real_data.size(0) < fake_data.size(0):
            idx = np.random.permutation(fake_data.size(0))[0: real_data.size(0)]
            fake_data = fake_data[idx, :]
        else:
            idx = np.random.permutation(real_data.size(0))[0: fake_data.size(0)]
            real_data = real_data[idx, :]

        if self.gp_alpha == "mix":
            alpha = torch.rand(real_data.size(0), 1).expand(real_data.size()).to(device)
            x_hat = alpha * real_data + (1 - alpha) * fake_data
        elif self.gp_alpha == "real":
            x_hat = real_data
        elif self.gp_alpha == "fake":
            x_hat = fake_data

        x_hat_out = self.discrim_net(x_hat.to(device).requires_grad_())
        gradients = torch.autograd.grad(outputs=x_hat_out, inputs=x_hat, \
                                        grad_outputs=torch.ones(x_hat_out.size()).to(device), \
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        if self.gp_lp:
            return (torch.max(0, gradients.norm(2, dim=1) - self.gp_center) ** 2).mean() * self.gp_lambda
        else:
            return ((gradients.norm(2, dim=1) - self.gp_center) ** 2).mean() * self.gp_lambda

    def behavior_cloning(self, policy_net=None, learning_rate=3e-4, bc_step=0):
        if bc_step <= 0 or policy_net is None:
            return

        bc_step_per_epoch = self.data_size / self.mini_batch_size
        bc_epochs = math.ceil(bc_step / bc_step_per_epoch)

        train = data_utils.TensorDataset(self.real_state_tensor.to(device), self.real_action_tensor.to(device))

        train_loader = data_utils.DataLoader(train, batch_size=self.mini_batch_size)
        optimizer_pi_bc = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

        count = 0
        print("Behavior cloning: %d epochs... " % bc_epochs)
        t0 = time.time()
        for epoch in range(0, bc_epochs):
            # for batch_idx, (s_batch, a_batch, w_batch) in enumerate(train_loader):
            for batch_idx, (s_batch, a_batch) in enumerate(train_loader):
                count = count + 1

                action_mean, _, _ = policy_net(s_batch)
                loss = 0.5 * ((action_mean - a_batch) ** 2).mean()  ##

                optimizer_pi_bc.zero_grad()
                loss.backward()
                optimizer_pi_bc.step()

        t1 = time.time()
        print("Pi BC %s steps (%2.2fs). Final MSE loss %2.5f" % (colored(count, p_color), t1 - t0, loss.item()))


""" GAIL """
class GAIL(IRL):
    def __init__(self, state_dim, action_dim, args):
        super().__init__(state_dim, action_dim, args)
        self.bce_negative = args.bce_negative
        if self.bce_negative:
            self.label_real = 0
            self.label_fake = 1  # with BCEloss: E_{pi}[log D] + E_{ex}[log(1-D)] with D = sigmoid
        else:
            self.label_real = 1
            self.label_fake = 0  # with BCEloss: E_{ex}[log D] + E_{pi}[log(1-D)] with D = sigmoid

    def compute_reward(self, states, actions, next_states=None, masks=None):
        if self.bce_negative:
            return -F.logsigmoid(self.discrim_net.get_reward(states, actions))  # maximize expert label score.
        else:
            return -F.logsigmoid(-self.discrim_net.get_reward(states, actions)) # minimize agent label score.

    def update_discriminator(self, batch, index, total_step=0, policy=None, writer=None, d=None):
        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(s_real, a_real)
        x_fake = self.discrim_net.get_reward(s_fake, a_fake)

        adversarial_loss = torch.nn.BCEWithLogitsLoss()
        label_real = Variable(FloatTensor(x_real.size(0), 1).fill_(self.label_real), requires_grad=False).to(device)
        label_fake = Variable(FloatTensor(x_fake.size(0), 1).fill_(self.label_fake), requires_grad=False).to(device)
        loss_real = adversarial_loss(x_real, label_real)
        loss_fake = adversarial_loss(x_fake, label_fake)
        loss = loss_real + loss_fake

        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()


class UID(IRL):
    def __init__(self, state_dim, action_dim, args):
        super().__init__(state_dim, action_dim, args)
        self.bce_negative = args.bce_negative
        if self.bce_negative:
            self.label_real = 0
            self.label_fake = 1  # with BCEloss: E_{pi}[log D] + E_{ex}[log(1-D)] with D = sigmoid
        else:
            self.label_real = 1
            self.label_fake = 0  # with BCEloss: E_{ex}[log D] + E_{pi}[log(1-D)] with D = sigmoid
        self.nn = args.nn

    def compute_reward(self, states, actions, next_states=None, masks=None):
        if self.bce_negative:
            return -F.logsigmoid(self.discrim_net.get_reward(states, actions))  # slightly different with paper
        else:
            return -F.logsigmoid(-self.discrim_net.get_reward(states, actions))  # minimize agent label score.


    def update_discriminator(self, batch, index, policy, writer, total_step=0, alpha=0.5, d=4):
        self.alpha = 0.7

        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(s_real, a_real)
        x_fake = self.discrim_net.get_reward(s_fake, a_fake)

        adversarial_loss = torch.nn.BCEWithLogitsLoss()
        label_real = Variable(FloatTensor(x_real.size(0), 1).fill_(self.label_real), requires_grad=False).to(device)
        label_fake = Variable(FloatTensor(x_fake.size(0), 1).fill_(self.label_fake), requires_grad=False).to(device)

        label_fake_real = Variable(FloatTensor(x_fake.size(0), 1).fill_(self.label_real), requires_grad=False).to(device)

        loss_fake = - adversarial_loss(x_fake, label_fake) * self.alpha  #  E_{pi}[log(1-D)] * alpha  with D = sigmoid

        loss_real = - adversarial_loss(x_real, label_real)  #  E_{ex}[log D]

        loss_fake_real_label = - adversarial_loss(x_fake, label_fake_real) * self.alpha  #  E_{pi}[log D] * alpha

        self.threshold = loss_real.detach() - loss_fake_real_label.detach()

        if self.nn:
            if self.threshold <= 0:
                loss = - (loss_fake + loss_real - loss_fake_real_label)
            else:
                loss = loss_real - loss_fake_real_label
        else:
            loss = - (loss_fake + loss_real - loss_fake_real_label)

        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()


""" UID for WAIL """
class UIDWAIL(IRL):
    def __init__(self, state_dim, action_dim, args, policy_updater=None):
        super().__init__(state_dim, action_dim, args)
        self.policy_updater = policy_updater
        self.label_real = 1
        self.label_fake = 0  # with BCEloss: E_{ex}[log D] + E_{pi}[log(1-D)] with D = sigmoid

    def compute_reward(self, states, actions, next_states=None, masks=None):
        return self.discrim_net.get_reward(states, actions)

    def update_discriminator(self, batch, index, writer, total_step=0):
        self.alpha = 0.7

        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(s_real, a_real)
        x_fake = self.discrim_net.get_reward(s_fake, a_fake)

        loss_real = x_real.mean()
        loss_fake = x_fake.mean()

        self.threshold = loss_real.detach() - self.alpha * loss_fake.detach()

        if self.threshold >= 0:
            loss = - (loss_real - 2 * self.alpha * loss_fake)
        else:
            loss = self.alpha * loss_fake - loss_real

        writer.add_scalar('threshold', self.threshold.cpu().detach().numpy(), total_step)

        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()


" D-REX and T-REX "
class REX():
    def __init__(self, state_dim, action_dim, args, initial_net=True):
        self.model_path = args.model_path
        self.state_dim = state_dim
        self.action_dim = action_dim

        """ trajectory description in result files """
        self.traj_name = "traj_type%d_N%d" % (args.c_data, args.demo_file_size)
        self.model_path = args.model_path
        print(self.model_path)
        if initial_net:
            self.initilize_nets(args, self.model_path)

        self.returns = None
        from my_utils.running_mean_std import RunningMeanStd
        self.rew_rms = RunningMeanStd(shape=())

    def initilize_nets(self, args, path):
        self.discrim_net = Reward(self.state_dim, self.action_dim).to(device)
        self.optimizer_discrim = torch.optim.Adam(self.discrim_net.parameters(), lr=args.learning_rate_d)
        self.discrim_net.load_state_dict(torch.load(path))

    def compute_reward(self, states, actions, next_states=None, masks=None, update_rms=True, gamma=0.99,
                       ctrl_coeff=0.001):
        traj = torch.cat((states, actions), dim=1)
        reward = self.discrim_net(traj)

        "previous"
        cliprew = 100.
        epsilon = 1e-8

        reward = reward.cpu().detach().numpy()
        self.rew_rms.update(reward)
        reward = np.clip((reward - self.rew_rms.mean) / np.sqrt(self.rew_rms.var + epsilon), -cliprew, cliprew)

        reward = reward - ctrl_coeff * np.sum(actions.cpu().detach().numpy() ** 2, axis=1).reshape(actions.shape[0], 1)
        rewards_tensor = torch.FloatTensor(reward)

        return rewards_tensor


" Two methods from ICML 19 < Imitation Learning from Imperfect Demonstrations > "
import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


class CULoss(nn.Module):
    def __init__(self, conf, beta, non=False):
        super(CULoss, self).__init__()
        self.loss = nn.SoftMarginLoss()
        self.beta = beta
        self.non = non
        if conf.mean() > 0.5:
            self.UP = True
        else:
            self.UP = False

    def forward(self, conf, labeled, unlabeled):
        y_conf_pos = self.loss(labeled, torch.ones(labeled.shape).to(device))
        y_conf_neg = self.loss(labeled, -torch.ones(labeled.shape).to(device))

        if self.UP:
            # conf_risk = torch.mean((1-conf) * (y_conf_neg - y_conf_pos) + (1 - self.beta) * y_conf_pos)
            unlabeled_risk = torch.mean(self.beta * self.loss(unlabeled, torch.ones(unlabeled.shape).to(device)))
            neg_risk = torch.mean((1 - conf) * y_conf_neg)
            pos_risk = torch.mean((conf - self.beta) * y_conf_pos) + unlabeled_risk
        else:
            # conf_risk = torch.mean(conf * (y_conf_pos - y_conf_neg) + (1 - self.beta) * y_conf_neg)
            unlabeled_risk = torch.mean(self.beta * self.loss(unlabeled, -torch.ones(unlabeled.shape).to(device)))
            pos_risk = torch.mean(conf * y_conf_pos)
            neg_risk = torch.mean((1 - self.beta - conf) * y_conf_neg) + unlabeled_risk
        if self.non:
            objective = torch.clamp(neg_risk, min=0) + torch.clamp(pos_risk, min=0)
        else:
            objective = neg_risk + pos_risk
        return objective


class PNLoss(nn.Module):
    def __init__(self):
        super(PNLoss, self).__init__()
        self.loss = nn.SoftMarginLoss()

    def forward(self, conf, labeled):
        y_conf_pos = self.loss(labeled, torch.ones(labeled.shape).to(device))
        y_conf_neg = self.loss(labeled, -torch.ones(labeled.shape).to(device))

        objective = torch.mean(conf * y_conf_pos + (1 - conf) * y_conf_neg)
        return objective


class Classifier(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.d1 = nn.Dropout(0.5)
        self.d2 = nn.Dropout(0.5)

        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.d1(torch.tanh(self.fc1(x)))
        x = self.d2(torch.tanh(self.fc2(x)))
        x = self.fc3(x)
        return x


""" 2IWIL """
class IWIL(IRL):
    def __init__(self, state_dim, action_dim, args):
        super().__init__(state_dim, action_dim, args)
        self.conf = torch.FloatTensor(args.conf)
        self.bce_negative = args.bce_negative
        if self.bce_negative:
            self.label_real = 0
            self.label_fake = 1  # with BCEloss: E_{pi}[log D] + E_{ex}[log(1-D)] with D = sigmoid
        else:
            self.label_real = 1
            self.label_fake = 0  # with BCEloss: E_{ex}[log D] + E_{pi}[log(1-D)] with D = sigmoid

        # def semi_classifier(self, ratio=0.2):
        ratio = 0.2
        num_label = int(ratio * self.conf.shape[0])
        p_idx = np.random.permutation(self.real_state_tensor.shape[0])
        state = self.real_state_tensor[p_idx, :]
        action = self.real_action_tensor[p_idx, :]
        conf = self.conf[p_idx, :]

        labeled_state = state[:num_label, :]
        labeled_action = action[:num_label, :]
        unlabeled_state = state[num_label:, :]
        unlabeled_action = action[num_label:, :]

        labeled_traj = torch.cat((labeled_state, labeled_action), dim=1).to(device)
        unlabeled_traj = torch.cat((unlabeled_state, unlabeled_action), dim=1).to(device)

        classifier = Classifier(labeled_state.shape[1] + labeled_action.shape[1], 40).to(device)
        classifier_optim = torch.optim.Adam(classifier.parameters(), 3e-4, amsgrad=True)
        cu_loss = CULoss(conf, beta=1 - ratio, non=True)

        batch = min(128, labeled_traj.shape[0])
        ubatch = int(batch / labeled_traj.shape[0] * unlabeled_traj.shape[0])
        iters = 25000

        print('Start Pre-train the Semi-supervised Classifier.')

        for i in range(iters):
            l_idx = np.random.choice(labeled_traj.shape[0], batch)
            u_idx = np.random.choice(unlabeled_traj.shape[0], ubatch)

            labeled = classifier(Variable(labeled_traj[l_idx, :]))
            unlabeled = classifier(Variable(unlabeled_traj[u_idx, :]))
            smp_conf = Variable(conf[l_idx, :].to(device))

            classifier_optim.zero_grad()
            risk = cu_loss(smp_conf, labeled, unlabeled)
            risk.backward()
            classifier_optim.step()

            if i % 1000 == 0:
                print('Iteration: {}\t cu_loss: {:.3f}'.format(i, risk.data.item()))

        classifier.cpu().eval()

        self.conf = torch.sigmoid(classifier(torch.cat((state, action), dim=1)))
        self.conf[:num_label, :] = conf[:num_label, :]

        self.real_state_tensor = state
        self.real_action_tensor = action

        self.conf.to(device)
        print('Confidence Prediction Ended.')

    def compute_reward(self, states, actions, next_states=None, masks=None):
        if self.bce_negative:
            return -F.logsigmoid(self.discrim_net.get_reward(states, actions))  # maximize expert label score.
        else:
            return -F.logsigmoid(-self.discrim_net.get_reward(states, actions))  # minimize agent label score.

    def update_discriminator(self, batch, index, total_step=0):
        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(s_real, a_real)
        x_fake = self.discrim_net.get_reward(s_fake, a_fake)

        adversarial_loss = torch.nn.BCEWithLogitsLoss()

        " Weighted GAIL Loss"
        p_mean = torch.mean(self.conf)
        p_value = self.conf[index, :] / p_mean

        weighted_loss = torch.nn.BCEWithLogitsLoss(weight=p_value.detach().to(device))
        label_real = Variable(FloatTensor(x_real.size(0), 1).fill_(self.label_real), requires_grad=False).to(device)
        label_fake = Variable(FloatTensor(x_fake.size(0), 1).fill_(self.label_fake), requires_grad=False).to(device)
        loss_real = weighted_loss(x_real, label_real)
        loss_fake = adversarial_loss(x_fake, label_fake)
        loss = loss_real + loss_fake

        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()


""" IC-GAIL """
class ICGAIL(IRL):
    def __init__(self, state_dim, action_dim, args):
        super().__init__(state_dim, action_dim, args)
        self.conf = torch.FloatTensor(args.conf)
        self.bce_negative = args.bce_negative
        if self.bce_negative:
            self.label_real = 0
            self.label_fake = 1  # with BCEloss: E_{pi}[log D] + E_{ex}[log(1-D)] with D = sigmoid
        else:
            self.label_real = 1
            self.label_fake = 0  # with BCEloss: E_{ex}[log D] + E_{pi}[log(1-D)] with D = sigmoid

        ratio = 0.2
        num_label = int(ratio * self.conf.shape[0])
        p_idx = np.random.permutation(self.real_state_tensor.shape[0])
        state = self.real_state_tensor[p_idx, :]
        action = self.real_action_tensor[p_idx, :]
        conf = self.conf[p_idx, :]

        labeled_state = state[:num_label, :]
        labeled_action = action[:num_label, :]
        unlabeled_state = state[num_label:, :]
        unlabeled_action = action[num_label:, :]

        self.labeled_conf = conf[:num_label, :]
        self.labeled_traj = torch.cat((labeled_state, labeled_action), dim=1)
        self.unlabeled_traj = torch.cat((unlabeled_state, unlabeled_action), dim=1)

        self.Z = torch.mean(conf[:num_label, :])
        self.Z = max(self.Z, float(0.7))

    def compute_reward(self, states, actions, next_states=None, masks=None):
        if self.bce_negative:
            return -F.logsigmoid(self.discrim_net.get_reward(states, actions))  # maximize expert label score.
        else:
            return -F.logsigmoid(-self.discrim_net.get_reward(states, actions))  # minimize agent label score.

    def update_discriminator(self, batch, index, total_step=0):

        idx = np.random.choice(self.unlabeled_traj.shape[0], int(0.8 * np.stack(batch.state).shape[0]))
        unlabeled = self.unlabeled_traj[idx, :].to(device)

        l_idx = np.random.choice(self.labeled_traj.shape[0], int(0.2 * np.stack(batch.state).shape[0]))
        labeled = self.labeled_traj[l_idx, :].to(device)
        labeled_conf = self.labeled_conf[l_idx, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(unlabeled[:, :self.state_dim], unlabeled[:, self.state_dim:])
        x_fake = self.discrim_net.get_reward(s_fake, a_fake)
        x_real_label = self.discrim_net.get_reward(labeled[:, :self.state_dim], labeled[:, self.state_dim:])

        " IC-GAIL "
        adversarial_loss = torch.nn.BCEWithLogitsLoss()
        weighted_loss = torch.nn.BCEWithLogitsLoss(weight=(1 - labeled_conf))

        label_real = Variable(FloatTensor(x_real.size(0), 1).fill_(self.label_real), requires_grad=False).to(device)
        label_fake = Variable(FloatTensor(x_fake.size(0), 1).fill_(self.label_fake), requires_grad=False).to(device)
        real_fake = Variable(FloatTensor(x_real_label.size(0), 1).fill_(self.label_fake), requires_grad=False).to(
            device)
        loss_real = adversarial_loss(x_real, label_real)
        loss_fake = adversarial_loss(x_fake, label_fake) * self.Z
        loss_weight = weighted_loss(x_real_label, real_fake) * (1 - self.Z) / (1 - labeled_conf.mean())
        loss = loss_real + loss_fake + loss_weight

        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((unlabeled[:, :self.state_dim], unlabeled[:, self.state_dim:]), 1),
                                    torch.cat((s_fake, a_fake)[:int(0.8 * np.stack(batch.state).shape[0])], 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()


class PUGAIL(IRL):
    def __init__(self, state_dim, action_dim, args):
        super().__init__(state_dim, action_dim, args)
        self.bce_negative = args.bce_negative
        if self.bce_negative:
            self.label_real = 0
            self.label_fake = 1  # with BCEloss: E_{pi}[log D] + E_{ex}[log(1-D)] with D = sigmoid
        else:
            self.label_real = 1
            self.label_fake = 0  # with BCEloss: E_{ex}[log D] + E_{pi}[log(1-D)] with D = sigmoid

        self.relu = nn.ReLU()

    def compute_reward(self, states, actions, next_states=None, masks=None):
        if self.bce_negative:
            return -F.logsigmoid(self.discrim_net.get_reward(states, actions))  # maximize expert label score.
        else:
            return -F.logsigmoid(-self.discrim_net.get_reward(states, actions))  # minimize agent label score.

    def update_discriminator(self, batch, index, policy, writer, total_step=0, alpha=0.5, d=4):

        self.alpha = 0.5

        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(s_real, a_real)
        x_fake = self.discrim_net.get_reward(s_fake, a_fake)

        adversarial_loss = torch.nn.BCEWithLogitsLoss()
        label_real = Variable(FloatTensor(x_real.size(0), 1).fill_(self.label_real), requires_grad=False).to(device)
        label_fake = Variable(FloatTensor(x_fake.size(0), 1).fill_(self.label_fake), requires_grad=False).to(device)

        label_real_fake = Variable(FloatTensor(x_real.size(0), 1).fill_(self.label_fake), requires_grad=False).to(
            device)

        loss_fake = - adversarial_loss(x_fake, label_fake)  # E_{pi}[log (1-D)]   with D = sigmoid

        loss_real = - adversarial_loss(x_real, label_real) * self.alpha # E_{ex}[log D] * alpha

        loss_real_fake_label = - adversarial_loss(x_real, label_real_fake) * self.alpha  # E_{e}[log (1-D)] * alpha

        self.threshold = loss_fake.detach() - loss_real_fake_label.detach()

        if self.threshold <= 0:
            loss = - (loss_real + loss_fake - loss_real_fake_label)
        else:
            loss = loss_fake - loss_real_fake_label

        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()
