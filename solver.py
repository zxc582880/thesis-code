# from originalmodel import  Generator
# from originalmodel import Discriminator ##原始模型

# from selfattmodel import  Generator
# from selfattmodel import Discriminator ##原始自注意力

# from proposemodel import  Generator
# from proposemodel import Discriminator ##proposed

# from SPAmodel import Generator
# from SPAmodel import Discriminator ##空間注意力

# from SEmodel import Generator
# from SEmodel import Discriminator ##Squeeze-and-Excitation

# from CAmodel import Generator
# from CAmodel import Discriminator ##通道注意力

from abstudy import Generator
from abstudy import Discriminator ##消融實驗

###############################################################################

from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
from PIL import Image
import datetime
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_skimage
import requests
import json
D_loss_list = []
D_loss_fake_list = []
D_loss_cls_list = []
D_loss_gp_list = []
G_loss_fake_list = []
G_loss_rec_list = []
G_loss_cls_list = []
G_gram_matrices = []
extra_G_loss_rec_list=[]




class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader
        config.use_attention = True  # 或者 False，根據你的需求


        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.G_gram_matrices = []  # 初始化 G_gram_matrices
        self.extra_G_loss_rec_list = []

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs
        self.outlier_penalty = 0.0
        

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step
        self.use_attention = config.use_attention

        

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
            # self.A = AttentionBlock()
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def train(self):
        """Train StarGAN within a single dataset."""

        # self.train_model(G_gram_matrices=self.G_gram_matrices)
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        def senddiscordmessage(webhookurl, message):
                    headers = {
                        'Content-Type': 'application/json'
                    }
                    payload = {
                        'content': message
                    }
                    response = requests.post(webhookurl, headers=headers, data=json.dumps(payload))
                    if response.status_code == 204:
                        print("Discord message sent successfully!")
                    else:
                        print("Failed to send Discord message. Status code:", response.status_code)
        local_time = time.localtime(time.time())
        hour_new = str(local_time.tm_hour).zfill(2)
        minute_new = str(local_time.tm_min).zfill(2)
        webhook_url = 'https://discord.com/api/webhooks/1223973029881712722/H3usCWUZ6VQQFqP9FUjUh-En3fw5sLCxaAtrVvbVCwvkiOn0cTYllUwwPyw16T_hIr6K'
        start_message = f"{hour_new}:{minute_new}。程式已開始"
        senddiscordmessage(webhook_url, start_message)
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            D_loss_list.append(loss['D/loss_real'])
            D_loss_fake_list.append(loss['D/loss_fake'])
            D_loss_cls_list.append(loss['D/loss_cls'])
            D_loss_gp_list.append(loss['D/loss_gp'])
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
           
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))
                

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                G_loss_fake_list.append(loss['G/loss_fake'])
                G_loss_rec_list.append(loss['G/loss_rec'])
                G_loss_cls_list.append(loss['G/loss_cls'])
            if (i+1) % 5000 ==0:
                loss['G/loss_rec_extra'] = g_loss_rec.item()
                extra_G_loss_rec_list.append(loss['G/loss_rec_extra'])
                # if self.use_attention:
                #     for attention_block in self.G.attention_blocks:
                #         G_gram_matrices.append(attention_block.gram_matrices)

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)

                # Print generator losses
                log += "Generator Losses:"
                for tag, value in loss.items():
                    log += ",{}: {:.4f}".format(tag, value)


                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))


            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
            
            if (i+1) % 600000 == 0:
                end_time=time.time()
                total_time=end_time-start_time
                print("總共{}秒".format(total_time))
                def senddiscordmessage(webhookurl, message):
                    headers = {
                        'Content-Type': 'application/json'
                    }
                    payload = {
                        'content': message
                    }
                    response = requests.post(webhookurl, headers=headers, data=json.dumps(payload))
                    if response.status_code == 204:
                        print("Discord message sent successfully!")
                    else:
                        print("Failed to send Discord message. Status code:", response.status_code)
                 # Discord Webhook URL
                local_time = time.localtime(time.time())
                hour_new = str(local_time.tm_hour).zfill(2)
                minute_new = str(local_time.tm_min).zfill(2)
                webhook_url = 'https://discord.com/api/webhooks/1223973029881712722/H3usCWUZ6VQQFqP9FUjUh-En3fw5sLCxaAtrVvbVCwvkiOn0cTYllUwwPyw16T_hIr6K'
                end_message = f"{hour_new}:{minute_new}。程式已結束"
                senddiscordmessage(webhook_url, end_message)
                file_path1 = os.path.join(self.result_dir, 'all_G_loss_rec_list.txt')
                with open(file_path1, 'w') as file:
                    for value in G_loss_rec_list:
                        file.write(f'{value}\n')
                
                avg_D_loss = np.mean(D_loss_list)
                avg_D_loss_fake = np.mean(D_loss_fake_list)
                avg_D_loss_cls = np.mean(D_loss_cls_list)
                avg_D_loss_gp = np.mean(D_loss_gp_list)
                avg_G_loss_fake = np.mean(G_loss_fake_list)
                avg_G_loss_rec = np.mean(G_loss_rec_list)
                avg_G_loss_cls = np.mean(G_loss_cls_list)
                var_D_loss = np.var(D_loss_list)
                var_D_loss_fake = np.var(D_loss_fake_list)
                var_D_loss_cls = np.var(D_loss_cls_list)
                var_D_loss_gp = np.var(D_loss_gp_list)
                var_G_loss_fake = np.var(G_loss_fake_list)
                var_G_loss_rec = np.var(G_loss_rec_list)
                var_G_loss_cls = np.var(G_loss_cls_list)
            

                
                

                # Print the average losses
                # print('Average D/loss: {:.4f}'.format(avg_D_loss))
                # print('Average D/loss_fake: {:.4f}'.format(avg_D_loss_fake))
                # print('Average D/loss_cls: {:.4f}'.format(avg_D_loss_cls))
                # print('Average D/loss_gp: {:.4f}'.format(avg_D_loss_gp))
                # print('Average G/loss_fake: {:.4f}'.format(avg_G_loss_fake))
                print('Average G/loss_rec: {:.4f}'.format(avg_G_loss_rec))
                # print('Average G/loss_cls: {:.4f}'.format(avg_G_loss_cls))
                # print('Variance D/loss: {:.4f}'.format(var_D_loss))
                # print('Variance D/loss_fake: {:.4f}'.format(var_D_loss_fake))
                # print('Variance D/loss_cls: {:.4f}'.format(var_D_loss_cls))
                # print('Variance D/loss_gp: {:.4f}'.format(var_D_loss_gp))
                # print('Variance G/loss_fake: {:.4f}'.format(var_G_loss_fake))
                print('Variance G/loss_rec: {:.4f}'.format(var_G_loss_rec))
                # print('Variance G/loss_cls: {:.4f}'.format(var_G_loss_cls))
                file_path2 = os.path.join(self.result_dir, 'loss.txt')
                with open(file_path2, 'w') as file:
                    file.write('共:'+f'{total_time}秒\n')
                    file.write('avg_loss:'+f'{avg_G_loss_rec}\n')
                    file.write('var_loss'+f'{var_G_loss_rec}\n')

           
    
    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        with torch.no_grad():
            start_time=time.time()
            for i, (x_real, c_org) in enumerate(data_loader):

                # Use the style from the first category for x_real
                x_real = x_real[0].unsqueeze(0).to(self.device)
                c_org = c_org[0].unsqueeze(0).to(self.device)

                # Prepare target domain labels for other categories
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                for j, c_trg in enumerate(c_trg_list):
                    # print('x_real size:', x_real.size())
                    # print('c_trg size:', c_trg.size())
                    # input

                    # Translate image.
                    x_fake = self.G(x_real, c_trg)

                    # Save the translated image.
                    result_path = os.path.join(self.result_dir, '{}-{}_image.jpg'.format(i+1, j+1))
                    save_image(self.denorm(x_fake.data.cpu()), result_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(result_path))
                    end_time=time.time()
            total_time=end_time-start_time
            print("總共{}秒".format(total_time))