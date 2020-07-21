import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from maze_dataset import MazeDataset
from unet_models import (
    AdditiveConditionalUNet,
    ConcatConditionalUNet,
)


dataloaders = {}
for mode in ['train', 'valid', 'test']:
    dataset = MazeDataset(
        root_dir='./maze_data_new',
        mode=mode,
        in_memory=False,
    )
    
    dataloaders[mode] = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=128, 
        shuffle=True, 
    )
    
    print('Num batches %s: %d' % (mode, len(dataloaders[mode])))


def print_result(last_img, true_img, pred_img, action, epoch, batch_idx, msg, title='idk.png'):
    frame_idx = np.random.choice(curr_states.shape[0])
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    fig.tight_layout()
   
    print('last img shape', last_img.shape)
    print('true img shape', true_img.shape)
    print('pred img shape', pred_img.shape)
 
    axes[0].imshow(last_img.detach().cpu().numpy())
    axes[1].imshow(true_img.detach().cpu().numpy()[0])            
    axes[2].imshow(pred_img.detach().cpu().numpy()[0])

    axes[0].set_title('{} Last seen frame, action {} '.format(msg, action))
    axes[1].set_title('{} True next frame'.format(msg))
    axes[2].set_title('{} Pred next frame epoch {} batch {}'.format(msg, epoch, batch_idx))

    plt.savefig('rgb-clean-predictions-unet/{}-action-{}-epoch-{}-batch-{}.png'.format(msg, action, epoch, batch_idx))
    plt.savefig(title)
    plt.show()
    plt.close()


def gradient_loss(gen_frames, gt_frames, alpha=1, reduction='mean'):

    def gradient(x):
        h_x = x.size()[-2]
        w_x = x.size()[-1]

        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        dx, dy = right - left, bottom - top 
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    # gradient
    gen_dx, gen_dy = gradient(gen_frames)
    gt_dx, gt_dy = gradient(gt_frames)
    #
    grad_diff_x = torch.abs(gt_dx - gen_dx)
    grad_diff_y = torch.abs(gt_dy - gen_dy)

    # condense into one tensor and avg
    if reduction == 'mean':
        return torch.mean(grad_diff_x ** alpha + grad_diff_y ** alpha)    
    elif reduction == 'none':
        return grad_diff_x ** alpha + grad_diff_y ** alpha


# ## Train a U-Net model
predict_difference=False
checkpoint_every = 30
show_results_every = 1
num_epochs = 10
factor = 1

train_batch_losses = []
val_batch_losses = []

idx_to_direction = {0: 'N', 1: 'S', 2: 'E', 3: 'W'}

criterion = torch.nn.L1Loss(reduction='none')

for unet_model in [ AdditiveConditionalUNet(n_channels=4, n_classes=1).cuda() ]:
    for lr in np.linspace(0.0007, 0.01, 10):
        for gdl_factor in [1, 2]:
            
            optimizer = torch.optim.Adam(unet_model.parameters(), lr=lr)
            
            print('Unet model {}, lr {}, gdl_factor {}'.format(
                type(unet_model).__name__, lr, gdl_factor))
            print('-' * 60)
            
            for epoch in range(num_epochs):
                for (batch_idx, batch) in enumerate(dataloaders['train']):
                    curr_states = batch['curr_state'].float().cuda()
                    actions = batch['action'].float().cuda()
                    next_frames = batch['next_frame'].float().cuda()
                    weights = batch['weights'].float().cuda()

                    output = torch.tanh(unet_model(curr_states, actions))
                    l1_loss = criterion(output, next_frames)
                    gdl_loss = gradient_loss(output, 
                                             next_frames, 
                                             reduction='none')
                    loss = factor * ((l1_loss + gdl_factor * gdl_loss) * weights).mean()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if batch_idx % checkpoint_every == 0:
                        checkpoint_name = 'models/rgb_model_{}_lr_{}_gdl_{}_ep_{}_batch_{}_loss_{}'.format(
                            type(unet_model).__name__, lr, gdl_factor, epoch, batch_idx, loss.item())
                        torch.save(
                            unet_model.state_dict(), 
                            checkpoint_name
                        )

                    if batch_idx % show_results_every == 0:
                        train_batch_losses.append(loss.item())
                        print('[train] Epoch {}, Batch {}, Loss {}'.format(
                            epoch, batch_idx, loss.item()))

                        frame_idx = np.random.choice(curr_states.shape[0])
                        action = idx_to_direction[torch.argmax(actions[frame_idx]).item()]
                        last_img = curr_states[frame_idx, -1]
                        true_img = next_frames[frame_idx]
                        pred_img = output[frame_idx]

                        if predict_difference:
                            true_img += curr_states[frame_idx, 3]
                            pred_img += curr_states[frame_idx, 3]

                        title = 'clean-predictions-unet/rgb_train_model_{}_lr_{}_gdl_{}_ep_{}_batch_{}_loss_{}.png'.format(
                            type(unet_model).__name__, lr, gdl_factor, epoch, batch_idx, loss.item())
                        print_result(last_img, 
                                     true_img, 
                                     pred_img, 
                                     action, 
                                     epoch, 
                                     batch_idx, 
                                     'train',
                                     title)

                        with torch.no_grad():
                            for val_batch in dataloaders['valid']:

                                val_curr_states = val_batch['curr_state'].float().cuda()
                                val_actions = val_batch['action'].float().cuda()
                                val_next_frames = val_batch['next_frame'].float().cuda()
                                val_weights = val_batch['weights'].float().cuda()

                                val_output = torch.tanh(unet_model(val_curr_states, 
                                                                   val_actions))
                                val_l1_loss = criterion(val_output, val_next_frames)
                                gdl_loss = gradient_loss(val_output, 
                                                         val_next_frames, 
                                                         reduction='none')
                                val_loss = factor * ((val_l1_loss + gdl_factor * gdl_loss) * val_weights).mean()
                                val_batch_losses.append(val_loss.item())

                                print('[val] Epoch {}, Batch {}, Loss {}'.format(
                                    epoch, batch_idx, val_loss.item()))

                                frame_idx = np.random.choice(val_curr_states.shape[0])
                                val_last_img = val_curr_states[frame_idx, -1]
                                val_action = idx_to_direction[torch.argmax(val_actions[frame_idx]).item()]
                                val_true_img = val_next_frames[frame_idx] 
                                val_pred_img = val_output[frame_idx]

                                if predict_difference:
                                    val_true_img += val_curr_states[frame_idx, 3]
                                    val_pred_img += val_curr_states[frame_idx, 3]

                                val_title = 'clean-predictions-unet/rgb_valid_model_{}_lr_{}_gdl_{}_ep_{}_batch_{}_loss_{}.png'.format(
                                    type(unet_model).__name__, lr, gdl_factor, epoch, batch_idx, loss.item())
                                print_result(val_last_img, 
                                             val_true_img, 
                                             val_pred_img, 
                                             val_action, 
                                             epoch, 
                                             batch_idx, 
                                             'valid',
                                             val_title)

                                break

                        plt.plot(np.arange(len(train_batch_losses)), 
                                 train_batch_losses, label='train')
                        plt.plot(np.arange(len(val_batch_losses)), 
                                 val_batch_losses, label='validation')
                        plt.legend(loc="upper right")
                        plt.savefig('rgb_loss.png'); plt.close()

