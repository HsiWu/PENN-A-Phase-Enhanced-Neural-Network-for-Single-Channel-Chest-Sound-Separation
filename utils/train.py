import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
import utils.Module as Module
import utils.MyData as MyData
import utils.MyData_test as MyData_test
from torch.utils.data import DataLoader
import os



def train(root_dir, cut_off_heart, cut_off_lung, model, path, complex_mask: bool, pure_phase: bool,
          dual_decoder:bool, d_model: int = 64, num_layers: int = 3, nhead: int = 16,
          step_each: int = 800, step_num: int = 200,
          length_time: int = 256, length_fre: int = 51, record: bool = False, classify=None):
    batch_size = 1
    learning_rate = 0.001
    # dim_feedforward: int = 128,
    warmup_step = step_num // 25 + 1
    warmup_lr_init = learning_rate / 100
    gamma = 0.01 ** (1 / (step_num - warmup_step))
    test_num = step_each
    os.makedirs('outcomes', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cut_off_lung != '40Hz' or cut_off_lung != '60Hz' or cut_off_lung != '80Hz' or cut_off_lung != '100Hz' \
            or cut_off_heart != '200Hz' or cut_off_heart != '225Hz' \
            or cut_off_heart != '250Hz' or cut_off_heart != '275Hz':
        RuntimeError('The cut-off frequency you entered does not exist')

    dataset_train = MyData.My_Data(root_dir, os.path.join('heart', 'train', cut_off_heart),
                                   os.path.join('lung', 'train', cut_off_lung),
                                   complex_mask=complex_mask, steps=step_num * step_each)
    dataset_test = MyData_test.My_Data(root_dir, os.path.join('heart', 'test', cut_off_heart),
                                       os.path.join('lung', 'test', cut_off_lung),
                                       complex_mask=complex_mask, steps=test_num, classify=classify)

    if model == 'Dual_Trans':
        the_model = Module.DualTrans(int(1 * d_model), num_layers, nhead, int(1 * d_model) * 2, path, device=device,
                                     complex_mask=complex_mask, pure_phase=pure_phase).to(device)
        model = path + '_Trans'
    elif model == 'Dual_Trans_Decoder':
        the_model = Module.DualTrans_Decoder(d_model, num_layers, nhead, d_model * 2, path, device=device,
                                             dual_decoder=dual_decoder,
                                     complex_mask=complex_mask, pure_phase=pure_phase).to(device)
        model = path + '_Trans_Decoder'
    else:
        RuntimeError('The mode that you entered does not exist')

    print(model + '-' + f'{complex_mask}' + '-' + cut_off_heart + '-' + cut_off_lung)
    num_params = sum(p.numel() for p in the_model.parameters() if p.requires_grad)
    print(num_params)

    # for name, param in the_model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.numel()}")

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(the_model.parameters(), lr=warmup_lr_init, eps=1e-7, weight_decay=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

    mask_np = np.ones((length_time, length_time), dtype=bool)
    for i in range(length_time):
        for j in range(length_time):
            if abs(i - j) <= length_time // 2:
                mask_np[i][j] = False
    mask_time = torch.tensor(data=mask_np, dtype=torch.bool).to(device)

    mask_np = np.ones((length_fre, length_fre), dtype=bool)
    for i in range(length_fre):
        for j in range(length_fre):
            if abs(i - j) <= length_fre:
                mask_np[i][j] = False
    mask_fre = torch.tensor(data=mask_np, dtype=torch.bool).to(device)
    # the_pool = nn.AvgPool2d((2, 2), padding=(0, 0))

    loss_fn1 = nn.MSELoss()

    loss_fn2 = nn.SmoothL1Loss(beta=0.01)



    list_of_train_loss = np.zeros((2, step_num))
    list_of_test_loss = np.zeros((2, step_num))
    total_train_step = 0
    count_train = 0
    train_loss = np.zeros(2)
    for data in dataloader_train:
        if complex_mask:
            heart_real, heart_imag, lung_real, lung_imag, \
            mask_real_heart, mask_imag_heart, mask_real_lung, mask_imag_lung, \
            mix_real, mix_imag = data

            mask_signals = torch.concatenate((mask_real_heart, mask_imag_heart, mask_real_lung, mask_imag_lung), dim=0)
        else:
            heart_real, heart_imag, lung_real, lung_imag, \
            mask_heart, mask_lung, mix_real, mix_imag = data

            mask_signals = torch.concatenate((mask_heart, mask_lung), dim=0)

        mix_real = mix_real.to(torch.float).to(device)
        mix_imag = mix_imag.to(torch.float).to(device)
        mask_signals = mask_signals.to(torch.float).to(device)
        heart_real = heart_real.to(torch.float).to(device)
        heart_imag = heart_imag.to(torch.float).to(device)
        lung_real = lung_real.to(torch.float).to(device)
        lung_imag = lung_imag.to(torch.float).to(device)
        mix = torch.concatenate((mix_real, mix_imag), dim=0)
        target = torch.concatenate((heart_real, heart_imag, lung_real, lung_imag), dim=0)

        # mask_heart = the_pool(mask_heart.to(device))

        the_model.train()
        mask_esti, signal_esti = the_model(src=mix, mask_time=mask_time, mask_fre=mask_fre)
        optimizer.zero_grad()

        loss1 = loss_fn1(mask_esti, mask_signals)
        loss2 = loss_fn2(signal_esti, target)
        if model == 'Dual_Trans_Decoder' or model == 'Time_Trans_Decoder' or model == 'Fre_Trans_Decoder':
            loss = 0.5 * loss1 / loss1.detach() + 0.5 * loss2 / loss2.detach()
        elif model == 'Dual_Trans' or model == 'Time_Trans' or model == 'Fre_Trans':
            loss = loss1 / loss1.detach()
        else:
            RuntimeError('No such model')

        loss.backward()
        optimizer.step()

        if record is True and 0 <= total_train_step % step_each < test_num:
            with torch.no_grad():
                count_train += 1
                train_loss[0] += loss1
                train_loss[1] += loss2
                if count_train == test_num:
                    train_loss = train_loss / test_num
                    print('train_loss', train_loss)
                    list_of_train_loss[:, total_train_step // step_each] = train_loss
                    count_train = 0
                    train_loss = np.zeros(2)

        if total_train_step % step_each == 0:
            step = total_train_step // step_each
            if step == 401:
                break

            if step <= warmup_step:
                warmup_lr = warmup_lr_init + (learning_rate - warmup_lr_init) * step / warmup_step
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            else:
                scheduler.step()

            if record is True or step % 10 == 0:
                test_loss = np.zeros(2)
                count = 0
                for data_test in dataloader_test:
                    if count == test_num:
                        break
                    count += 1

                    if complex_mask:
                        heart_real, heart_imag, lung_real, lung_imag, \
                        mask_real_heart, mask_imag_heart, mask_real_lung, mask_imag_lung, \
                        mix_real, mix_imag = data_test
                        mask_signals = torch.concatenate(
                            (mask_real_heart, mask_imag_heart, mask_real_lung, mask_imag_lung),
                            dim=0)
                    else:
                        heart_real, heart_imag, lung_real, lung_imag, \
                        mask_heart, mask_lung, mix_real, mix_imag = data_test

                        mask_signals = torch.concatenate((mask_heart, mask_lung), dim=0)

                    mix_real = mix_real.to(torch.float).to(device)[:, :51, :]
                    mix_imag = mix_imag.to(torch.float).to(device)[:, :51, :]
                    mask_signals = mask_signals.to(torch.float).to(device)[:, :51, :]
                    heart_real = heart_real.to(torch.float).to(device)[:, :51, :]
                    heart_imag = heart_imag.to(torch.float).to(device)[:, :51, :]
                    lung_real = lung_real.to(torch.float).to(device)[:, :51, :]
                    lung_imag = lung_imag.to(torch.float).to(device)[:, :51, :]
                    mix = torch.concatenate((mix_real, mix_imag), dim=0)
                    target = torch.concatenate((heart_real, heart_imag, lung_real, lung_imag), dim=0)

                    # mask_heart = the_pool(mask_heart.to(device))
                    with torch.no_grad():
                        the_model.eval()
                        mask_esti, signal_esti = the_model(src=mix, mask_time=mask_time, mask_fre=mask_fre)
                        test_loss[0] += loss_fn1(mask_esti, mask_signals)
                        test_loss[1] += loss_fn2(signal_esti, target)

                test_loss = test_loss / test_num
                list_of_test_loss[:, step] = test_loss
                print('step', step, 'test_loss', test_loss)
        total_train_step += 1

    torch.save({'model_state_dict': the_model.state_dict()},
               os.path.join('models', 'model' + '-' + model + '-' + f'{complex_mask}' + '-' + f'{pure_phase}' +
                            '-' + f'{dual_decoder}' + '-' + cut_off_heart + '-' + cut_off_lung))
    torch.save(list_of_train_loss,
               os.path.join('outcomes', 'train_loss' + '-' + model + '-' + f'{complex_mask}' + '-' + f'{pure_phase}' +
                           '-' + f'{dual_decoder}' + cut_off_heart + '-' + cut_off_lung))
    torch.save(list_of_test_loss,
               os.path.join('outcomes', 'test_loss' + '-' + model + '-' + f'{complex_mask}' + '-' + f'{pure_phase}' +
                           '-' + f'{dual_decoder}' + cut_off_heart + '-' + cut_off_lung))
