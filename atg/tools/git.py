import re
import shlex
import subprocess
from pathlib import Path
from subprocess import PIPE
from typing import Optional

from rich.console import Console
from rich.theme import Theme

custom_theme = Theme(
    {
        "def": "white",
        "pchg": "bold deep_pink1",
        "prem": "reverse cyan",
        "pname": "chartreuse1",
        "rname": "light_goldenrod2",
        "bname": "white",
        "fup": "light_goldenrod2",
        "rmto": "deep_sky_blue3",
        "cto": "violet",
        "cinfo": "deep_sky_blue3",
        "cstate": "deep_pink1",
    }
)

console = Console(theme=custom_theme)


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
            console.print(f"     |--[cstate]{c[0]}[/cstate][fup] {c[1]}[/fup][def]")

    if show_stash:
        stashed = get_stashed(rep)
        if len(stashed):
            print("  |--Stashed")
            for num, s in enumerate(stashed):
                console.print(f"     |-- [cstate]{num}[/cstate][def]{s[0]} {s[2]}")

    def to_push_to_pull(rep: str, branch: str, to_function, to_str=str):
        remotes = get_remote_repositories(rep)
        for r in remotes:
            commits = to_function(rep, r, branch)
            if len(commits) > 0:
                print(f"  |--{r}")
                for commit in commits:
                    li = f"[cto][{to_str}][/cto] [cinfo]{commit}[/cinfo][def]"
                    console.print(f"     |--{li}")

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
                to_return += (
                    f" [rname]{r}[/rname][def][rmto]{to_str}[/rmto][def]:{count}[/def]"
                )

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
            pname = f"[pchg]{repname}[/pchg][def]"
        elif not bool(remotes):
            pname = f"[prem]{repname}[/prem][def]"
        else:
            pname = f"[pname]{repname}[/pname][def]"

        if islocal:
            strlocal = f"[rname]Local[/rname][def][rmto] To Commit:[/rmto][def]{len(changes)}[/def]"
        else:
            strlocal = ""

        console.print(f"{pname}/[bname]{branch}[/bname] {strlocal}{topush}{topull}")

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
        console.print("\a")
