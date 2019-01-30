import pickle
import os
    

def analyze_expert_data():
    filename = '/home/ext_drive/somilb/data/expert_data/sbpd/sbpd_projected_grid/' \
               'sbpd_area6/200_goals_10_seed/expert_success_data.pkl'
    
    # Load the file
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        
    import ipdb; ipdb.set_trace()
    print('debugging')


if __name__ == '__main__':
    analyze_expert_data()
