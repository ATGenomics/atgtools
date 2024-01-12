import re
import shlex
import subprocess
from pathlib import Path
from subprocess import PIPE
from typing import Optional

from colored import attr, fg

CT = {
    "def": attr("reset") + fg("white"),
    "pchg": attr("reset") + attr("bold") + fg("deep_pink_1a"),
    "prem": attr("reverse") + fg("light_cyan"),
    "pname": attr("reset") + fg("chartreuse_1"),
    "rname": attr("reset") + fg("light_goldenrod_2b"),
    "bname": attr("reset") + fg("white"),
    "fup": attr("reset") + fg("light_goldenrod_2b"),
    "rmto": attr("reset") + fg("deep_sky_blue_3b"),
    "cto": attr("reset") + fg("violet"),
    "cinfo": attr("reset") + fg("deep_sky_blue_3b"),
    "cstate": attr("reset") + fg("deep_pink_1a"),
    "bell": "\a",
    "reset": "\033[2J\033[H",
}


def search_repositories(path_dir: Optional[str]) -> list:
    did = Path.cwd() if path_dir is None else Path(path_dir).resolve()
    repos = [x.parent.as_posix() for x in did.glob("**/.git")]
    return repos


def git_exec(path: str, cmd: str):
    command_line = f"git -C {path} {cmd}"
    cmdargs = shlex.split(command_line)
    p = subprocess.Popen(cmdargs, stdout=PIPE, stderr=PIPE)
    output, errors = p.communicate()
    if p.returncode:
        print(f"Failed running {command_line}")
        raise Exception(errors)
    return output.decode("utf-8")


def update_remote(rep: str):
    git_exec(rep, "remote update")


def get_branches(rep: str, only_default: bool = True):
    gitbranch = git_exec(f"{rep}", "branch")

    if only_default:
        sbranch = re.compile(r"^\* (.*)", flags=re.MULTILINE)
        return {m.group(1) if (m := sbranch.search(gitbranch)) else ""}

    branch = gitbranch.splitlines()
    return [b[2:] for b in branch]


def get_local_files_change(rep: str, checkuntracked: bool):
    snbchange = re.compile(r"^(.{2}) (.*)")
    result = git_exec(rep, f"status -s{'' if checkuntracked else 'uno'}")
    lines = result.split("\n")
    return [[m.group(1), m.group(2)] for x in lines if (m := snbchange.match(x))]


def get_remote_repositories(rep):
    result = git_exec(f"{rep}", "remote")
    remotes = [x for x in result.split("\n") if x]
    return remotes


def has_remote_branch(rep, remote, branch):
    result = git_exec(rep, "branch -r")
    return f"{remote}/{branch}" in result


def get_local_to_push(rep, remote, branch):
    if not has_remote_branch(rep, remote, branch):
        return []
    result = git_exec(rep, f"log {remote}/{branch}..{branch} --oneline")
    return [x for x in result.split("\n") if x]


def get_remote_to_pull(rep, remote, branch):
    if not has_remote_branch(rep, remote, branch):
        return []
    result = git_exec(rep, f"log {branch}..{remote}/{branch} --oneline")

    return [x for x in result.split("\n") if x]


def verbosity(changes, show_stash: bool, rep: str, branch: str):
    if len(changes) > 0:
        print("  |--Local")
        for c in changes:
            print(f"     |--{CT['cstate']}{c[0]}{CT['fup']} {c[1]}{CT['def']}")
    if show_stash:
        stashed = get_stashed(rep)
        if len(stashed):
            print("  |--Stashed")
            for num, s in enumerate(stashed):
                print(f"     |-- {CT['cstate']}{num}{CT['def']}{s[0]} {s[2]}")

    def to_push_to_pull(rep: str, branch: str, to_function, to_str=str):
        remotes = get_remote_repositories(rep)
        for r in remotes:
            commits = to_function(rep, r, branch)
            if len(commits) > 0:
                print(f"  |--{r}")
                for commit in commits:
                    li = f"{CT['cto']}[{to_str}]{CT['def']} {CT['cinfo']}{commit}{CT['def']}"
                    print(f"     |--{li}")

    if branch != "":
        to_push_to_pull(
            rep=rep, branch=branch, to_function=get_local_to_push, to_str="To Push"
        )
        to_push_to_pull(
            rep=rep, branch=branch, to_function=get_remote_to_pull, to_str="To Pull"
        )


# Check state of a git repository
def check_repository(
    rep: str, branch: str, show_stash, checkuntracked: bool, quiet: bool, verbose: bool
):
    changes = get_local_files_change(rep, checkuntracked)
    islocal = len(changes) > 0

    if show_stash:
        islocal = islocal or len(get_stashed(rep)) > 0

    ischange = islocal
    action_needed = False
    topush = topull = ""
    repname = remotes = None

    def count_topush_topull(
        rep: str,
        branch: str,
        remotes: list,
        to_function,
        to_str: str,
        to_return: str,
        ischange: bool,
        action_needed: bool,
    ):
        for r in remotes:
            count = len(to_function(rep, r, branch))
            ischange = ischange or (count > 0)
            action_needed = action_needed or (count > 0)

            if count > 0:
                to_return += f" {CT['rname']}{r}{CT['def']}[{CT['rmto']}{to_str}:{CT['def']}{count}]"

            return to_return, ischange, action_needed

    if branch != "":
        remotes = get_remote_repositories(rep)
        topush, ischange, action_needed = count_topush_topull(
            rep=rep,
            branch=branch,
            remotes=remotes,
            to_function=get_local_to_push,
            to_str="To Push",
            to_return=topush,
            ischange=ischange,
            action_needed=action_needed,
        )

        topull, ischange, action_needed = count_topush_topull(
            rep=rep,
            branch=branch,
            remotes=remotes,
            to_function=get_remote_to_pull,
            to_str="To Pull",
            to_return=topull,
            ischange=ischange,
            action_needed=action_needed,
        )

    if ischange or not quiet:
        if rep == str(Path.cwd()):
            repname = Path.cwd().name
        elif rep.find(str(Path.cwd())) == 0:
            repname = rep[len(str(Path.cwd())) + 1 :]
        else:
            repname = rep

        if ischange:
            pname = f"{CT['pchg']}{repname}{CT['def']}"
        elif not bool(remotes):
            pname = f"{CT['prem']}{repname}{CT['def']}"
        else:
            pname = f"{CT['pname']}{repname}{CT['def']}"

        if islocal:
            strlocal = f"{CT['rname']}Local{CT['def']}[{CT['rmto']}To Commit:{CT['def']}{len(changes)}]"
        else:
            strlocal = ""

        print(f"{pname}/{CT['bname']}{branch} {strlocal}{topush}{topull}")

        if verbose:
            verbosity(changes, show_stash, rep, branch)

    return action_needed


def get_stashed(rep):
    result = git_exec(rep, "stash list --oneline")

    split_lines = [x.split(" ", 2) for x in result.split("\n") if x]

    return split_lines


# Check all git repositories
def gitcheck(
    verbose: bool,
    checkremote: bool,
    checkuntracked: bool,
    bell_on_action_needed: bool,
    search_dir: str,
    quiet: bool,
    checkall: str,
    show_stash: bool,
):
    repo = search_repositories(path_dir=search_dir)
    action_needed = False

    if checkremote:
        for r in repo:
            print(f"Updating \033[1m{Path(r).name}\033[0m remotes...")
            update_remote(r)

    for r in repo:
        if checkall:
            branch = get_branches(r, only_default=False)
        else:
            branch = get_branches(r)

        for b in branch:
            if check_repository(
                rep=r,
                branch=b,
                show_stash=show_stash,
                checkuntracked=checkuntracked,
                quiet=quiet,
                verbose=verbose,
            ):
                action_needed = True

    if action_needed and not bell_on_action_needed:
        print(CT["bell"])
