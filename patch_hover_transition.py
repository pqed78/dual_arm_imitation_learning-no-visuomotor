import re

def patch_file(filename, is_batched=False):
    with open(filename, 'r') as f:
        text = f.read()
        
    lines = text.split('\n')
    in_hover = False
    for i, line in enumerate(lines):
        if 'elif phase == PHASE_LEFT_HOVER_TARGET:' in line:
            in_hover = True
        elif 'elif phase == PHASE_LEFT_LOWER_TARGET:' in line:
            in_hover = False
            
        if in_hover and 'if dist' in line and 'phase_timer >' in line:
            if not is_batched:
                # Add dist_xy and dist_z computation right before the if statement
                lines.insert(i, "                dist_xy = torch.norm(tgt_hover[:, :2] - tcp_pos_l[:, :2])")
                lines.insert(i+1, "                dist_z = torch.abs(tgt_hover[:, 2] - tcp_pos_l[:, 2])")
                # modify line which is now at i+2
                lines[i+2] = "                if (dist_xy < 0.01 and dist_z < 0.015) or phase_timer > 70:"
            else:
                lines.insert(i, "                dist_xy = torch.norm(tgt_hover[:, :2] - tcp_pos_l[:, :2], dim=-1)")
                lines.insert(i+1, "                dist_z = torch.abs(tgt_hover[:, 2] - tcp_pos_l[:, 2])")
                lines[i+2] = "                if (dist_xy.max().item() < 0.01 and dist_z.max().item() < 0.015) or phase_timer > 70:"
            break

    with open(filename, 'w') as f:
        f.write('\n'.join(lines))

patch_file('scripts/generate_scripted_demos.py', is_batched=False)
patch_file('scripts/generate_scripted_demos2.py', is_batched=True)
