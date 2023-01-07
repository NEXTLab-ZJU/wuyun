def train(train_loader, model, optimizer,epoch,total_epoch):
    # define the format of tqdm
    with tqdm(total=len(train_loader), ncols=150) as _tqdm: 
        _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, total_epoch)) 

        # Model Train
        model.train()
        running_loss = 0.0
        total_token_loss =0.0
        total_dur_loss =0.0
        total_vel_loss = 0.0
        for idx, data in enumerate(train_loader):
            # get the inputs;
            enc_inputs = {k: data[f'cond_{k}'].cuda() for k in CON_KEYS}
            dec_inputs = {k: data[f'tgt_{k}'].cuda() for k in TGT_KEYS}
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize |
            token_out, vel_out, dur_out, tempo_out = model(enc_inputs, dec_inputs)

            # 1) token_out
            target = (data['tgt_token'].cuda())[:, 1:]
            token_loss = xe_loss(token_out[:, :-1], target)
            # 2) vel_out
            if hparams['use_velocity']:
                tgt_vel = data['tgt_vel'][:, 1:]
                tgt_vel = tgt_vel * (tgt_vel != 31).long()  # skip padding velocity
                vel_loss = xe_loss(vel_out[:, :-1], tgt_vel.cuda()) * hparams['lambda_attr']
            else:
                vel_loss = 0
            # 3) dur_loss
            dur_loss = xe_loss(dur_out[:, :-1], (data['tgt_dur'][:, 1:]).cuda()) * hparams['lambda_attr']

            # 8) tempo loss
            tgt_tempo = (data['tgt_tempo'].cuda())[:, 1:]
            tempo_loss = xe_loss(tempo_out[:, :-1], tgt_tempo)

            total_loss = token_loss + vel_loss + dur_loss + tempo_loss
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            total_token_loss+=token_loss.item()
            total_dur_loss+=dur_loss.item()
            total_vel_loss+=vel_loss.item()

            _tqdm.set_postfix(loss="{:.6f}, token_loss ={:.6f}, dur_loss={:.6f}, vel_loss={:.6f}".format(running_loss,total_token_loss,total_dur_loss,total_vel_loss))
            _tqdm.update(1)

    return running_loss