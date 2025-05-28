import torch
import numpy as np
import logging

from functions.utils.motion_simulation.motion_helpers import expand_mps_to_kspline_resolution

def sim_motion_get_gt_motion_traj(args, traj, verbose=True):

    if args.motionTraj_simMot == "uniform_interShot_event_model":
        logging.info(f"Generate inter-shot random motion parameters with seed {args.random_motion_seed}, motion states {args.Ns}, number of shots {args.num_shots}, max translation/rotation {args.max_trans}/{args.max_rot}, num motion events {args.num_motion_events}") if verbose else None
        gt_motion_params = gen_rand_mot_params_interShot(args.Ns, args.max_trans, args.max_rot, args.random_motion_seed, args.num_motion_events, args.num_shots)

        gt_motion_params,_,_ = expand_mps_to_kspline_resolution(gt_motion_params, traj, list_of_track_dc_losses=None)

        traj_updated = ([np.array([k]) for k in traj[0][0]], [np.array([k]) for k in traj[1][0]])
        for i in torch.arange(1,args.Ns):
            # For each shot expand the traj to per line resolution
            traj_updated[0].extend([np.array([k]) for k in traj[0][i]])
            traj_updated[1].extend([np.array([k]) for k in traj[1][i]])

        intraShot_event_inds = None
    elif args.motionTraj_simMot == "uniform_interShot_event_model_simreal":
        logging.info(f"Generate inter-shot random motion parameters with seed {args.random_motion_seed}, motion states {args.Ns}, number of shots {args.num_shots}, max translation/rotation {args.max_trans}/{args.max_rot}, num motion events {args.num_motion_events}") if verbose else None
        gt_motion_params = gen_motion_params_simreal(args.Ns, args.num_motion_events, args.main_mot_range, args.perturb_mot_range,args.random_motion_seed)
                        
        gt_motion_params,_,_ = expand_mps_to_kspline_resolution(gt_motion_params, traj, list_of_track_dc_losses=None)

        traj_updated = ([np.array([k]) for k in traj[0][0]], [np.array([k]) for k in traj[1][0]])
        for i in torch.arange(1,args.Ns):
            # For each shot expand the traj to per line resolution
            traj_updated[0].extend([np.array([k]) for k in traj[0][i]])
            traj_updated[1].extend([np.array([k]) for k in traj[1][i]])

        intraShot_event_inds = None
    elif args.motionTraj_simMot == "uniform_intraShot_event_model":
        logging.info(f"Generate intra-shot random motion parameters with seed {args.random_motion_seed}, motion states {args.Ns}, number of shots {args.num_shots}, max translation/rotation {args.max_trans}/{args.max_rot}, num motion events {args.num_motion_events}, num intraShot events {args.num_intraShot_events}") if verbose else None
        gt_motion_params, traj_updated, intraShot_event_inds = gen_rand_mot_params_intraShot(args.Ns, args.max_trans, args.max_rot, 
                                                                                                          args.random_motion_seed, args.num_motion_events, 
                                                                                                          args.num_intraShot_events, args.num_shots, traj)

        logging.info(f"Number of motion states in traj without intra-shot motion {len(traj[0])} and with intra-shot motion {len(traj_updated[0])}") if verbose else None
    else:
        raise ValueError(f"motionTraj_simMot {args.motionTraj_simMot} not implemented.")

    return gt_motion_params.cuda(args.gpu), traj_updated, intraShot_event_inds

def sample_non_adjacent_numbers(n=3, total=52, min_gap=4):
    while True:
        nums = torch.randperm(total-min_gap)[:n]
        sorted_nums = nums.sort()[0]
        if all((sorted_nums[i+1] - sorted_nums[i]) >= min_gap for i in range(n - 1)):
            return sorted_nums

def gen_motion_params_simreal(Ns, num_events, main_mot_range, perturb_mot_range,seed):
    """
    Generate motion parameters for a given number of events and severity.
    Args:
        Ns (int): Number of shots.
        num_events (int): Number of motion events.
        main_mot_range (float): Main motion range.
        perturb_mot_range (float): Perturbation motion range.
    Returns:
        motion_params (torch.Tensor): Generated motion parameters.
    """
    torch.manual_seed(seed)
    # Define the instruction list:
    instruction_list = ["nod","shake","turn_left_right_stay","turn_left_right_return","chin_down_stay","chin_down_return"]
    max_instruction_shot = 4
    motion_params = torch.zeros(Ns, 6)
    # Randomly pick which states will trigger an event (excluding the very first one for variety)
    event_start_shots = sample_non_adjacent_numbers(num_events,Ns,max_instruction_shot)

    for i in range(num_events):
        # Perturbations on other motion parameters:
        if i == len(event_start_shots)-1:
            motion_params[event_start_shots[i]:,[0, 1, 2, 5]] = torch.rand(4) * 2 * perturb_mot_range - perturb_mot_range
        else:
            motion_params[event_start_shots[i]:event_start_shots[i+1],[0, 1, 2, 5]] = torch.rand(4) * 2 * perturb_mot_range - perturb_mot_range
        # Main motion parameters:
        picked_instruction = instruction_list[torch.randint(0, len(instruction_list), (1,)).item()]
        # picked_instruction = "chin_down_return"
        # print(picked_instruction)
        if picked_instruction == "nodding":
            motion_params[event_start_shots[i], 4] = 0
            motion_params[event_start_shots[i]+1, 4] = -1 * torch.rand(1) * main_mot_range
            motion_params[event_start_shots[i]+2, 4] = torch.rand(1) * main_mot_range
            motion_params[event_start_shots[i]+3, 4] = 0
        elif picked_instruction == "shake":
            motion_params[event_start_shots[i], 3] = 0
            motion_params[event_start_shots[i]+1, 3] = -1 * torch.rand(1) * main_mot_range
            motion_params[event_start_shots[i]+2, 3] = torch.rand(1) * main_mot_range
            motion_params[event_start_shots[i]+3, 3] = 0
        elif picked_instruction == "turn_left_right_stay":
            motion_params[event_start_shots[i], 3] = 0
            if i == len(event_start_shots)-1:
                motion_params[event_start_shots[i]+1:, 3] = torch.rand(1) * 2 * main_mot_range - main_mot_range
            else:
                motion_params[event_start_shots[i]+1:event_start_shots[i+1]:, 3] = torch.rand(1) * 2 * main_mot_range - main_mot_range
        elif picked_instruction == "turn_left_right_return":
            motion_params[event_start_shots[i], 3] = 0
            motion_params[event_start_shots[i]+1, 3] = torch.rand(1) * 2 * main_mot_range - main_mot_range
            motion_params[event_start_shots[i]+2, 3] = 0
        elif picked_instruction == "chin_down_stay":
            motion_params[event_start_shots[i], 4] = 0
            if i == len(event_start_shots)-1:
                motion_params[event_start_shots[i]+1:, 4] = torch.rand(1) * 2 * main_mot_range - main_mot_range
            else:
                motion_params[event_start_shots[i]+1:event_start_shots[i+1]:, 4] = torch.rand(1) * 2 * main_mot_range - main_mot_range
        elif picked_instruction == "chin_down_return":
            motion_params[event_start_shots[i], 4] = 0
            motion_params[event_start_shots[i]+1, 4] = torch.rand(1) * 2 * main_mot_range - main_mot_range
            motion_params[event_start_shots[i]+2, 4] = 0
    return motion_params

def gen_rand_mot_params_interShot(Ns, max_trans, max_rot, seed, num_events, num_shots):
    '''
    Draw num_events many out of Ns motion states that receive a unique patient position.
    The remaining motion states get the patient position of the last event.
    One position is defined by 3 translations and 3 rotations, which are drawn uniformly
    from [-max_rot/-max_trans, max_rot/max_trans]
    Input:
        - Ns: number of motion states
        - max_trans: maximum translation in pixels
        - max_rot: maximum rotation in degrees
        - seed: random seed
        - num_events: number of motion states with unique patient positions
        - num_shots: number of shots
    Output:
        - motion_params: tensor of shape (Ns, 6) with the motion parameters    
        - motion_params_shots_states_map: array with one entry per shot containing the number of motion states in this shot
    '''
    assert Ns == num_shots, "Number of motion states must match number of shots for inter-shot motion simulation."
    motion_params = torch.zeros(Ns, 6)
    motion_params_events = torch.zeros(num_events, 6)
    torch.manual_seed(seed)
    motion_params_events[:,0:3] = torch.rand([num_events,3]) * 2 * max_trans - max_trans
    motion_params_events[:,3:6] = torch.rand([num_events,3]) * 2 * max_rot - max_rot

    # pick random motion states for the events and sort them
    # remove the zero entry from event_states (we assume not motion during the first shot)
    event_states = torch.randperm(Ns)
    event_states = event_states[event_states != 0]
    event_states = event_states[:num_events]
    
    event_states = torch.sort(event_states)[0]
    for i in range(len(event_states)):
        if i == len(event_states)-1:
            motion_params[event_states[i]:,:] = motion_params_events[i:i+1,:]
        else:
            motion_params[event_states[i]:event_states[i+1],:] = motion_params_events[i:i+1,:]

    return motion_params

def gen_rand_mot_params_intraShot(Ns, max_trans, max_rot, seed, num_events, num_intraShot_events, num_shots, traj):
    '''
    Draw num_events many out of Ns motion states that receive a unique patient position.
    The remaining motion states get the patient position of the last event.
    One position is defined by 3 translations and 3 rotations, which are drawn uniformly
    from [-max_rot/-max_trans, max_rot/max_trans].
    num_intraShot_events-many events receive a motion state per line within this shot.
    Those states linearly interpolate following and preceeding motion parameters.
    Input:
        - Ns: number of motion states
        - max_trans: maximum translation in pixels
        - max_rot: maximum rotation in degrees
        - seed: random seed
        - num_events: number of shots with unique patient positions
        - num_intraShot_events: number of motion shots with one motion state per line
        - num_shots: number of shots. Must be equal to Ns
        - traj: tuple, where traj[0]/traj[1] contains a list of Ns-many x/y-coordinates
    Output:
        - motion_params: tensor of shape (number of k-space lines, 6) with ground truth the motion parameters    
        - traj_updated: tuple, where traj_updated[0]/traj_updated[1] contains a list of number-of-k-space-lines-many x/y-coordinates
        - events_intrashot: list of shot indices with intra-shot motion
    '''
    assert Ns == num_shots, "Number of motion states must match number of shots for intra-shot motion simulation."
    assert num_intraShot_events <= num_events, "Number of intra-shot events must be smaller or equal to the number of events."

    # Generate inter-shot motion parameters (one set of 6 parameters for each shot following the event model)
    motion_params_inter = torch.zeros(Ns, 6)
    motion_params_events = torch.zeros(num_events, 6)
    #torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    motion_params_events[:,0:3] = torch.rand([num_events,3], generator=gen) * 2 * max_trans - max_trans
    motion_params_events[:,3:6] = torch.rand([num_events,3], generator=gen) * 2 * max_rot - max_rot

    # pick random motion shots for where the motion events take place and sort them (exclude first shot)
    event_states_shuffled = torch.arange(1,Ns)
    event_states_shuffled = event_states_shuffled[torch.randperm(len(event_states_shuffled), generator=gen)][:num_events]
    event_states = torch.sort(event_states_shuffled)[0]

    # Assign generated motion parameters to the shots following the event model
    for i in range(len(event_states)):
        if i == len(event_states)-1:
            motion_params_inter[event_states[i]:,:] = motion_params_events[i:i+1,:]
        else:
            motion_params_inter[event_states[i]:event_states[i+1],:] = motion_params_events[i:i+1,:]

    # from the event shots randomly pick the ones that are intra-shot motion events
    events_intrashot = event_states_shuffled[:num_intraShot_events]
    logging.info(f"Shot indices with intra-shot motion: {events_intrashot}")
    
    # In this model the first shot has constant all-zero motion parameters
    motion_params = torch.zeros(len(traj[0][0]), 6)
    traj_updated = ([np.array([k]) for k in traj[0][0]], [np.array([k]) for k in traj[1][0]])

    for i in torch.arange(1,num_shots):
        # For each shot expand the traj to per line resolution
        traj_updated[0].extend([np.array([k]) for k in traj[0][i]])
        traj_updated[1].extend([np.array([k]) for k in traj[1][i]])
        
        if i in events_intrashot:
            if i == Ns-1:
                # There is a intra-shot event in the last shot. 
                # Concatenate random motion state to motion_params_inter to compute ending points of the intra-shot motion
                pass
            # Design intra-shot motion
            num_lines = len(traj[0][i])

            # Decide whether motion has already started during break
            if torch.rand(1, generator=gen) < 0.3:
                # motion starts withing this shot, somewhere in the first third of the shot
                num_starting_lines_const = int(torch.rand(1, generator=gen) / 3 * num_lines)
            else:
                # motion started before
                num_starting_lines_const = 0

            # Decide whether motion ends during that shot
            if torch.rand(1, generator=gen) < 0.3:
                # motion ends withing this shot, somewhere in the last third of the shot
                num_ending_lines_const = int(torch.rand(1, generator=gen) / 3 * num_lines)
            else:
                # motion continues
                num_ending_lines_const = 0

            num_lines_motion = num_lines - num_starting_lines_const - num_ending_lines_const
            motion_params_intra = torch.zeros(num_lines, 6)

            for j in range(6):
                
                # if motion starts or ends during the shot add motion parameters of the previous or next shot accordingly
                motion_params_intra[:num_starting_lines_const,j] = motion_params_inter[i-1,j]
                motion_params_intra[num_lines-num_ending_lines_const:,j] = motion_params_inter[i,j]

                shot_gap = torch.abs(motion_params_inter[i-1,j] - motion_params_inter[i,j]).item()

                if num_starting_lines_const == 0:
                    # generate offset at start of shot with random fraction of the shot gap and random sign
                    offset_fraction = torch.rand(1, generator=gen).item() / 5
                    offset_sign = torch.sign(torch.rand(1, generator=gen) - 0.5).item()
                    starting_point = motion_params_inter[i-1,j].item() + offset_sign * offset_fraction * shot_gap
                else:
                    starting_point = motion_params_inter[i-1,j].item()

                if num_ending_lines_const == 0:
                    # generate offset at end of shot with random fraction of the shot gap and random sign
                    offset_fraction = torch.rand(1, generator=gen).item() / 5
                    offset_sign = torch.sign(torch.rand(1, generator=gen) - 0.5).item()
                    ending_point = motion_params_inter[i,j].item() - offset_sign * offset_fraction * shot_gap
                else:
                    ending_point = motion_params_inter[i,j].item()

                # determine the motion parameters for the lines with motion
                # determine number of peaks
                tmp = torch.rand(1, generator=gen)
                motion_ratio = num_lines_motion / num_lines
                if tmp < motion_ratio/3:
                    if tmp < motion_ratio/6:
                        # two peaks
                        if torch.rand(1, generator=gen) < 0.5:
                            # first overshoot towards the direction of the next motion state
                            ending_point_peak_1 = ending_point + max_rot*(0.1+torch.rand(1, generator=gen)/3).item()
                            ending_point_peak_2 = ending_point - max_rot*(0.1+torch.rand(1, generator=gen)/3).item()
                        else:
                            # first have peak in the opposite direction of the next motion state
                            ending_point_peak_1 = starting_point - max_rot*(0.1+torch.rand(1, generator=gen)/3).item()
                            ending_point_peak_2 = ending_point + max_rot*(0.1+torch.rand(1, generator=gen)/5).item()

                        motion_params_intra[num_starting_lines_const:num_starting_lines_const+num_lines_motion//3,j] = torch.linspace(starting_point, ending_point_peak_1, num_lines_motion//3)
                        motion_params_intra[num_starting_lines_const+num_lines_motion//3:num_starting_lines_const+2*(num_lines_motion//3),j] = torch.linspace(ending_point_peak_1, ending_point_peak_2, num_lines_motion//3)
                        motion_params_intra[num_starting_lines_const+2*(num_lines_motion//3):num_starting_lines_const+num_lines_motion,j] = torch.linspace(ending_point_peak_2, ending_point, num_lines_motion-(2*(num_lines_motion//3)))
                    else:
                        # one peak
                        if torch.rand(1, generator=gen) < 0.5:
                            # overshoot towards the direction of the next motion state
                            ending_point_peak = ending_point + max_rot*(0.1+torch.rand(1, generator=gen)/3).item()
                        else:
                            # have peak in the opposite direction of the next motion state
                            ending_point_peak = starting_point - max_rot*(0.1+torch.rand(1, generator=gen)/3).item()
                        motion_params_intra[num_starting_lines_const:num_starting_lines_const+num_lines_motion//2,j] = torch.linspace(starting_point, ending_point_peak, num_lines_motion//2)
                        motion_params_intra[num_starting_lines_const+num_lines_motion//2:num_starting_lines_const+num_lines_motion,j] = torch.linspace(ending_point_peak, ending_point, num_lines_motion-num_lines_motion//2)
                else:
                    # zero peaks
                    motion_params_intra[num_starting_lines_const:num_starting_lines_const+num_lines_motion,j] = torch.linspace(starting_point, ending_point, num_lines_motion)

            motion_params = torch.cat((motion_params, motion_params_intra), dim=0)
        else:
            # Repeat the motion parameters of this shot to obtain per line resolution
            motion_params = torch.cat((motion_params, motion_params_inter[i:i+1,:].repeat(len(traj[0][i]),1)), dim=0)

    return motion_params, traj_updated, events_intrashot