"""Run from the repository root with the audited local Python interpreter."""
import argparse
import json
from pathlib import Path
from .pipeline import config_from,audit,generate_prior
from .storage import plain,read_json
from .report import status,report


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("audit")
    p.add_argument("--config",required=True)
    p = sub.add_parser("run")
    p.add_argument("--config",required=True)
    p.add_argument("--profile",choices=("smoke","pilot","study"),default="smoke")
    p.add_argument("--output",required=True)
    p.add_argument("--budget-json")
    p.add_argument("--task-archive")
    p.add_argument("--checkpoint")
    p.add_argument("--prior-evidence")
    p.add_argument("--stages",choices=("generate","audit","all"),default="all")
    p.add_argument("--dry-run",action="store_true")
    for name in ("status","resume","report"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir",required=True)
        if name=="resume":
            p.add_argument("--dry-run",action="store_true")
            p.add_argument("--stages",choices=("generate","audit","all"),default="all")
    args = parser.parse_args(argv)
    if args.command=="audit":
        value = audit(config_from(args.config))[0]
    elif args.command=="run":
        config = config_from(args.config)
        if args.prior_evidence:
            config["explicit_scaleup_evidence"]=args.prior_evidence
        budget = dict(config["profiles"][args.profile])
        if args.budget_json and args.profile!="study":
            overrides = read_json(args.budget_json)
            if set(overrides)-set(budget):
                raise ValueError("unknown budget fields")
            budget.update(overrides)
        if args.profile=="study":
            # No automatic TEST population creation. A study needs a separately
            # frozen task archive and explicit physical resource authorization.
            if not all((args.task_archive,args.budget_json,args.checkpoint)):
                value = {"status":"BLOCKED","reason":"study requires --task-archive, --budget-json, --checkpoint and --prior-evidence"}
            else:
                from .study import run_study
                value=run_study(config,args.task_archive,args.budget_json,args.checkpoint,args.output,args.dry_run)
        elif args.dry_run:
            value = {"status":"DRY_RUN","profile":args.profile,"budget":budget,"output":args.output}
        else:
            value = generate_prior(config,budget,args.output,args.profile,args.stages)
    elif args.command=="resume":
        run = read_json(Path(args.run_dir)/"run.json")
        if run.get("kind")=="study":
            from .study import run_study
            value=run_study(run["config"],run["archive"],run["budget_path"],run["checkpoint"],args.run_dir,args.dry_run,True)
        else:
            value = {"status":"DRY_RUN","identity":run["identity"]} if args.dry_run else generate_prior(
                run["config"],run["identity"]["budget"],args.run_dir,run["identity"]["profile"],args.stages,True)
    else:
        runfile=Path(args.run_dir)/"run.json"
        if runfile.exists() and read_json(runfile).get("kind")=="study":
            result=Path(args.run_dir)/"study.json"
            value=read_json(result) if result.exists() else {"status":"IN_PROGRESS","ledger":read_json(Path(args.run_dir)/"ledger.json")}
        else:
            value = status(args.run_dir) if args.command=="status" else report(args.run_dir)
    print(json.dumps(value,default=plain,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
