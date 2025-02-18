### Before run any program, press L2 + R2 to get into debug mode !!! ### 

# RUN POLICY
# Press select button to quit the program at any time.
python <deploy.py | deploy_motion_lib.py> <network interface> <xxx.yaml>

# 1. Start button to move to default pose.
# 2. A button to eval the policy.


# G1 flat
python deploy.py <network interface> g1.yaml


# G1 Height Map
# Press B button to use the GT Height Map
python deploy.py <network interface> g1_height_map.yaml