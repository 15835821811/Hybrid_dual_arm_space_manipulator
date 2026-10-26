"""Prepare a separate dynamic unit, without freezing inputs or running physics."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/'ff_prior_terminal_tracking';NEW=ROOT/'ff_prior_horizon_tracking'
assert not (NEW/'job.py').exists(),'Dynamic implementation already prepared'
NEW.mkdir(exist_ok=True);(NEW/'docs').mkdir();(NEW/'tools').mkdir();(NEW/'evidence').mkdir()
for name in ('__init__.py','job.py','job.ps1','environment.py','evaluation.py','experiment.py','inputs.py','visuals.py','test_workflow.py'):
    text=(OLD/name).read_text(encoding='utf-8-sig').replace('ff_prior_terminal_tracking','ff_prior_horizon_tracking')
    if name=='inputs.py':
        text=text.replace("ROOT/'ff_prior_diffusion_demo/evidence/selection.json'","ROOT/'ff_prior_horizon_restore/evidence/selection.json'")
        text=text.replace("ROOT/'ff_prior_diffusion_demo/evidence/binding.json'","ROOT/'ff_prior_horizon_restore/evidence/binding.json'")
        text=text.replace("ROOT/'ff_prior_fullchain/evidence/diffusion_demo_closeout.json'","ROOT/'ff_prior_fullchain/evidence/horizon_restore_closeout.json'")
        text=text.replace("report=read_json(SOURCE/'units/u00/execution/report.json')",
            "closed=read_json(ROOT/'ff_prior_fullchain/evidence/horizon_restore_closeout.json')\n    assert closed['closed'] and closed['commit']\n    verdict=read_json(ROOT/closed['independent_review'])\n    assert verdict['C1']['claim_supported']=='yes','STATIC_REVIEW_NOT_CONFIRMED'\n    report=read_json(SOURCE/'units/u00/execution/report.json')")
        text=text.replace("inherited_D1_diffusion_generated_static_path; dynamic_controller_not_a_new_diffusion_model",
            "historical_Diffusion_guess_plus_full323_static_restoration; dynamic_feedback_not_a_new_network")
        text=text.replace("original_dynamic_DEV_claim=False","original_five_obstacles_preserved=True,original_dynamic_DEV_claim=False")
        text=text.replace("source_terminal_index=len(a['time'])-1,","source_terminal_index=len(a['time'])-1,source_static_review=closed['independent_review'],")
        text=text.replace("ROOT/'ff_prior_fullchain/evidence/horizon_restore_closeout.json'):",
            "ROOT/'ff_prior_fullchain/evidence/horizon_restore_closeout.json',ROOT/closed['independent_review'],\n              ROOT/'ff_prior_horizon_restore/evidence/authorization.json'):")
    if name=='experiment.py':
        text=text.replace("for tick in range(1350):","for tick in range(P['control_ticks']):").replace("total=1350,","total=P['control_ticks'],")
        text=text.replace("budget=Budget(out);budget.save();c=ident['contract']","budget=Budget(out);budget.save();c=ident['contract']")
        text=text.replace("budget.check();budget.pending", "budget.check('physics');budget.pending")
        text=text.replace("min(budget.deadline,time.time()+1200)","min(budget.deadline,time.time()+P['media_reserve_s'])")
        text=text.replace("summary['visualization']=generate(out,robot,c,progress,min(budget.deadline,time.time()+P['media_reserve_s']))",
            "try:summary['visualization']=generate(out,robot,c,progress,min(budget.deadline,time.time()+P['media_reserve_s']))\n        except Exception as error:summary['visualization']=dict(status='MEDIA_PARTIAL',error=repr(error),new_physics_steps=0)\n        budget.check('total');budget.save()")
        text=text.replace("summary=dict(status='FINISHED',","summary=dict(status='FINISHED' if passed else 'FINISHED_WITH_BLOCKED_ITEMS',")
    if name=='evaluation.py':
        text=text.replace("complete=len(ctrl)==P['steps_each'] and reason is None",
            "complete=(len(ctrl)==P['steps_each'] and reason is None and len(h['time'])==P['steps_each']+1\n        and abs(h['time'][0]-env.origin_time)<1e-10 and abs(h['time'][-1]-env.origin_time-P['duration'])<1e-9)")
        text=text.replace("finite=bool(np.isfinite(h['qpos']).all() and np.isfinite(h['qvel']).all())",
            "finite=bool(all(np.isfinite(v).all() for v in h.values()) and np.isfinite(ctrl).all()\n            and np.isfinite(raw).all() and np.isfinite(env.integration_history).all()))")
        # Replacement above includes the dict close exactly once.
        text=text.replace("np.isfinite(env.integration_history).all())))","np.isfinite(env.integration_history).all()))")
    if name=='visuals.py':
        text=text.replace("from ff_prior_diffusion_demo.visuals import savefig,HTML",
            "from ff_prior_horizon_restore.visuals import savefig,HTML\nfrom .media_geometry import add_saved_overlays,JS_OVERLAYS")
        text=text.replace("from .protocol import SOURCE","from .protocol import SOURCE")
        text=text.replace("renderer.update_scene(d,camera=camera,scene_option=option)",
            "renderer.update_scene(d,camera=camera,scene_option=option)\n                add_saved_overlays(renderer.scene,arrays['ee'],goals[k],int(k),indices)")
        text=text.replace("ee=full['ee'][indices,:,:3,3].tolist(),obstacles=",
            "ee=full['ee'][indices,:,:3,3].tolist(),ee_poses=full['ee'][indices].tolist(),goal_poses=goals[indices].tolist(),obstacles=")
        text=text.replace("html=HTML.replace('dot(D.goals[k]','dot(D.goals[i][k]')",
            "html=HTML.replace('dot(D.goals[k]','dot(D.goals[i][k]').replace('__OVERLAY_JS__',JS_OVERLAYS)")
        text=text.replace("固定三障碍、原世界目标与27秒运动。动画来自真实保存状态；动态连续交接和五障碍尚属后续阶段。系统通过不等于裸网络独立通过，不进行相对优势评判。",
            "原固定五障碍、原世界目标与27秒运动。动画来自实际保存状态；动态连续交接尚属后续阶段。系统通过不等于裸网络独立通过，不进行相对优势评判。")
        text=text.replace("卫星继承零速度；五障碍与移动卫星尚未通过。","原五障碍贯穿两段；卫星继承零速度，尚未证明实体漂移。")
        text=text.replace("完整静态接近＋实际终态接七不规则航点。","原五障碍完整静态接近＋实际终态接原七不规则航点。")
        text=text.replace("native_steps_for_visualization=0,combined_actual_duration_s=",
            "native_steps_for_visualization=0,overlays=dict(actual_ee_frames=True,time_varying_goal_frames=True,actual_ee_trajectories=True,axis_colors='RGB = XYZ'),combined_actual_duration_s=")
        # All static figures are retained alongside dynamic figures in the combined index.
        text=text.replace("videos=[];errors=[]","static_figures=[]\n    for source in sorted((SOURCE/'units/u00/visuals').glob('0*')):\n        if source.suffix in ('.png','.pdf'):\n            shutil.copy2(source,out/('static_'+source.name))\n            if source.suffix=='.png':static_figures.append('static_'+source.stem)\n    videos=[];errors=[]")
        text=text.replace("('01_dynamic_summary','02_joint_motion','03_tracking_momentum','04_world_paths')))",
            "['01_dynamic_summary','02_joint_motion','03_tracking_momentum','04_world_paths']+static_figures))")
        text=text.replace("previous=np.load(SOURCE/'units/u00/execution/native_trace.npz')",
            "previous=np.load(SOURCE/'units/u00/execution/native_trace.npz')\n    assert np.array_equal(previous['integration'][-1],a['integration'][0]),'MEDIA_HANDOFF_NOT_IDENTICAL'")
    (NEW/name).write_text(text,encoding='utf-8',newline='\n')
media=(ROOT/'ff_prior_horizon_restore/media_geometry.py').read_text(encoding='utf-8')
media=media.replace("label+' '+ 'XYZ'[axis]","label if axis==2 else ''")
media=media.replace("D.goal_poses[arm]","D.goal_poses[i][arm]")
(NEW/'media_geometry.py').write_text(media,encoding='utf-8')
print(NEW)
