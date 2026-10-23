from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/v7_1/moving_satellite_001'
SOURCE=ROOT/'artifacts/v7_1/balanced_brake_001'
P=dict(version='physical_satellite_origin_v1',duration=29.,burn_s=2.,coast_s=27.,dt=.002,
       steps_each=14500,control_ticks=1450,threads=2,wall_s=4800,media_reserve_s=1200,
       total_unit_wall_s=6000,implementation_reserve_s=1200,max_bytes=1024**3,
       output_reserve_bytes=192*1024**2,max_native_replay_steps=29000,max_coupled=2000,
       planned_coupled=0,executions=1,retries=0,seed=20261020,training=False,expansion=False,
       TEST=False,advantage_comparison=False,position_gate_m=.005,rotation_gate_deg=1.,
       geometry_step_m=.002,replay_atol=1e-10,body_diagnostics_period_s=.02,
       visualization_distance_band_m=.01,peak_force_N=.04,force_axis_world=[0.,0.,1.],
       force_frame='world; xfrc_applied force at body COM, zero COM torque',
       force_sampling='midpoint of each 2ms interval; zero for input indices >=1000',
       linear_balance_atol=1e-9,angular_balance_atol=1e-9,work_balance_atol_J=1e-9,
       analytic_position_atol_m=3e-6,analytic_velocity_atol_m_s=1e-8,
       coast_velocity_atol_m_s=1e-10,force_atol_N=1e-14,final_hold_s=1.5,
       max_recording_estimate_bytes=800*1024**2,max_active_rows=600000,
       body_distance_samples=1451,model_propagation_calls=0,
       preserve_world_waypoints=True,actual_state_clipping=False,velocity_injection=False)
