# -*- coding: utf-8 -*-
# Copyright (c) 2024 Manuel Schneider
#
# https://docker-py.readthedocs.io/en/stable/
#

import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import docker
from albert import *

md_iid = "5.0"
md_version = "4.1.3"
md_name = "Docker"
md_description = "Manage docker images, containers and compose projects"
md_license = "MIT"
md_url = "https://github.com/albertlauncher/albert-plugin-python-docker"
md_authors = ["@ManuelSchneid3r"]
md_maintainers = ["@ManuelSchneid3r"]
md_bin_dependencies = ["docker"]
md_lib_dependencies = ["docker"]

COMPOSE_PROJECT = "com.docker.compose.project"
COMPOSE_SERVICE = "com.docker.compose.service"
COMPOSE_FILES = "com.docker.compose.project.config_files"
COMPOSE_DIR = "com.docker.compose.project.working_dir"
COMPOSE_ENV = "com.docker.compose.project.environment_file"

PAST = {"start": "started", "up": "started", "stop": "stopped",
        "restart": "restarted", "remove": "removed", "pause": "paused", "unpause": "unpaused"}

# States a container must be in for pause/unpause, compose fails on a partially paused project
OP_STATES = {"pause": ("running",), "unpause": ("paused",)}

CACHE_TTL = 2.0  # seconds; a query fires per keystroke


def _short_image(image: str) -> str:
    """Untagged images show up as their digest."""
    return image[7:19] if image.startswith("sha256:") else image


@dataclass
class Container:
    id: str
    name: str
    image: str
    state: str   # running, exited, restarting, paused, created, dead
    status: str  # human readable, e.g. "Up 3 hours"
    project: str | None
    service: str | None

    @property
    def running(self) -> bool:
        return self.state in ("running", "restarting", "paused")


@dataclass
class Project:
    name: str
    files: list[str]
    working_dir: str | None
    env_files: list[str] = field(default_factory=list)
    containers: list[Container] = field(default_factory=list)

    @property
    def states(self) -> dict[str, int]:
        counts = {}
        for c in self.containers:
            counts[c.state] = counts.get(c.state, 0) + 1
        return counts

    @property
    def running(self) -> bool:
        return any(c.running for c in self.containers)


class Plugin(PluginInstance, RankedQueryHandler):
    # Global query handler not applicable, queries take seconds sometimes

    icon_blue = Path(__file__).parent / "running.svg"
    icon_gray = Path(__file__).parent / "stopped.svg"

    def __init__(self):
        PluginInstance.__init__(self)
        RankedQueryHandler.__init__(self)
        self.client = None
        self._cache = None
        self._cache_time = 0.0
        self._lock = threading.Lock()
        self._notifications = []  # keep references, a collected Notification is dismissed

    def synopsis(self, query):
        return "<project|container> | @<project> [container] | img <tag> | stop all"

    def defaultTrigger(self):
        return "d "

    # ---- icons ---------------------------------------------------------------------------------

    def _icon(self, running: bool, grapheme: str):
        return Icon.composed(Icon.image(self.icon_blue if running else self.icon_gray),
                             Icon.grapheme(grapheme))

    def _container_icon(self, c: Container):
        grapheme = {"restarting": "⚠️", "paused": "⏸️"}.get(c.state, "📦")
        return self._icon(c.state != "paused" and c.running, grapheme)

    def _project_icon(self, project: Project):
        if "restarting" in project.states:
            return self._icon(True, "⚠️")
        if "paused" in project.states and "running" not in project.states:
            return self._icon(False, "⏸️")
        return self._icon(project.running, "🗂️")

    # ---- data ----------------------------------------------------------------------------------

    def _fetch(self):
        """Returns (projects by name, containers). One API call, cached for CACHE_TTL."""
        with self._lock:
            if self._cache and time.monotonic() - self._cache_time < CACHE_TTL:
                return self._cache

            projects: dict[str, Project] = {}
            containers: list[Container] = []
            # Low level API: one request, no per container image lookup
            for raw in self.client.api.containers(all=True):
                labels = raw.get("Labels") or {}
                project = None
                if name := labels.get(COMPOSE_PROJECT):
                    project = projects.setdefault(name, Project(
                        name,
                        [f for f in labels.get(COMPOSE_FILES, "").split(",") if f],
                        labels.get(COMPOSE_DIR),
                        [f for f in labels.get(COMPOSE_ENV, "").split(",") if f]))

                c = Container(
                    id=raw["Id"],
                    name=raw["Names"][0].lstrip("/") if raw.get("Names") else raw["Id"][:12],
                    image=_short_image(raw.get("Image", "")),
                    state=raw.get("State", ""),
                    status=raw.get("Status", ""),
                    project=project.name if project else None,
                    service=labels.get(COMPOSE_SERVICE),
                )
                containers.append(c)
                if project:
                    project.containers.append(c)

            self._cache = (projects, containers)
            self._cache_time = time.monotonic()
            return self._cache

    def _invalidate(self):
        with self._lock:
            self._cache = None

    # ---- actions -------------------------------------------------------------------------------

    def _stop_all(self, containers):
        ids = [(c.id, c.name) for c in containers if c.running]

        def run():
            failed = []

            def stop(cid, name):
                try:
                    self.client.containers.get(cid).stop()
                except Exception as e:
                    failed.append(f"{name}: {e}")

            threads = [threading.Thread(target=stop, args=i) for i in ids]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self._invalidate()
            if failed:
                warning("; ".join(failed))
                self._notify(f"Docker: {len(failed)} of {len(ids)} failed to stop",
                             "\n".join(failed[:5]))
            else:
                self._notify("Docker", f"Stopped {len(ids)} containers")

        threading.Thread(target=run, daemon=True).start()

    def _stop_all_item(self, containers):
        running = [c for c in containers if c.running]
        return StandardItem(
            id="stop-all",
            text=f"Stop all running containers ({len(running)})",
            subtext="Projects and standalone containers",
            icon_factory=lambda: self._icon(False, "⏹️"),
            actions=[Action("stop-all", "Stop all", lambda: self._stop_all(running))],
        )

    def _notify(self, title, text):
        n = Notification(title, text)
        n.send()
        self._notifications = self._notifications[-4:] + [n]

    def _run_bg(self, name, op, fn):
        """Runs fn in a thread, invalidates the cache and notifies about the outcome."""
        def run():
            try:
                fn()
                self._notify(name, PAST.get(op, f"{op} done"))
            except Exception as e:
                warning(f"{name}: {op}: {e}")
                self._notify(f"{name}: {op} failed", str(e))
            finally:
                self._invalidate()
        threading.Thread(target=run, daemon=True).start()

    def _container_op(self, c: Container, op, **kwargs):
        return lambda: self._run_bg(c.name, op,
                                    lambda: getattr(self.client.containers.get(c.id), op)(**kwargs))

    @staticmethod
    def _compose_cmd(project: Project, *args, with_files=False) -> list[str]:
        """Without files compose works on the container labels only, so stop, restart and logs
        do not fail on missing variables or moved files. up needs the files and env files."""
        cmd = ["docker", "compose", "-p", project.name]
        if with_files:
            for f in project.files:
                cmd += ["-f", f]
            for f in project.env_files:
                cmd += ["--env-file", f]
            if project.working_dir:
                cmd += ["--project-directory", project.working_dir]
        return cmd + list(args)

    def _compose_op(self, project: Project, *args):
        cmd = self._compose_cmd(project, *args, with_files=args[0] == "up")

        def run():
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError(p.stderr.strip().splitlines()[-1] if p.stderr.strip()
                                   else f"exit code {p.returncode}")

        return lambda: self._run_bg(project.name, args[0], run)

    def _project_containers_op(self, project: Project, op):
        """Runs op on the project'project containers that are in a matching state."""
        ids = [c.id for c in project.containers if c.state in OP_STATES[op]]
        return lambda: self._run_bg(
            project.name, op,
            lambda: [getattr(self.client.containers.get(i), op)() for i in ids])

    @staticmethod
    def _terminal(script, cwd=None):
        if cwd:
            script = f"cd {shlex.quote(cwd)} && {script}"
        return lambda: runTerminal(f"{script} ; exec $SHELL")

    # ---- items ---------------------------------------------------------------------------------

    def _project_item(self, project: Project):
        states = ", ".join(f"{n} {st}" for st, n in sorted(project.states.items()))
        where = ", ".join(project.files) or project.working_dir or ""
        actions = []

        if project.running:
            actions += [Action("stop", "Stop project", self._compose_op(project, "stop")),
                        Action("restart", "Restart project", self._compose_op(project, "restart")),
                        Action("up", "Up -d (start missing services)",
                               self._compose_op(project, "up", "-d"))]
            if "running" in project.states:
                actions.append(Action("pause", "Pause project",
                                      self._project_containers_op(project, "pause")))
            if "paused" in project.states:
                actions.insert(0, Action("unpause", "Unpause project",
                                         self._project_containers_op(project, "unpause")))
        else:
            actions.append(Action("up", "Start project (up -d)",
                                  self._compose_op(project, "up", "-d")))
        actions.append(Action("logs", "Logs", self._terminal(
            shlex.join(self._compose_cmd(project, "logs", "-f", "--tail", "200")))))
        if project.files:
            actions.append(Action("copy-path", "Copy compose path",
                                  lambda p=",".join(project.files): setClipboardText(p)))
            actions.append(Action("open-file", "Open file", lambda f=project.files[0]: openFile(f)))
        if project.working_dir:
            actions.append(Action("terminal", "Open terminal in project directory",
                                  self._terminal("true", project.working_dir)))
            actions.append(Action("copy-dir", "Copy project directory",
                                  lambda d=project.working_dir: setClipboardText(d)))

        return StandardItem(
            id=f"project:{project.name}",
            text=project.name,
            subtext=f"{states} · {where}",
            icon_factory=lambda: self._project_icon(project),
            actions=actions,
            input_action_text=f"@{project.name} ",
        )

    def _container_item(self, c: Container, projects):
        if c.state == "paused":
            actions = [Action("unpause", "Unpause container", self._container_op(c, "unpause")),
                       Action("stop", "Stop container", self._container_op(c, "stop"))]
        elif c.running:
            actions = [Action("stop", "Stop container", self._container_op(c, "stop")),
                       Action("restart", "Restart container", self._container_op(c, "restart"))]
            if c.state == "running":
                actions.append(Action("pause", "Pause container", self._container_op(c, "pause")))
        else:
            actions = [Action("start", "Start container", self._container_op(c, "start"))]
        actions += [
            Action("logs", "Logs", self._terminal(f"docker logs -f --tail 200 {c.id}")),
            Action("shell", "Shell", self._terminal(
                f"docker exec -it {c.id} "
                "sh -c 'command -v bash >/dev/null && exec bash || exec sh'")),
        ]

        project = projects.get(c.project)
        if project and project.files:
            actions.append(Action("copy-path", "Copy compose path",
                                  lambda p=",".join(project.files): setClipboardText(p)))
        actions += [
            Action("copy-id", "Copy id to clipboard", lambda cid=c.id: setClipboardText(cid)),
            Action("remove", "Remove (forced, with volumes)",
                   self._container_op(c, "remove", v=True, force=True)),
        ]

        prefix = f"{c.project}/{c.service} · " if c.project else ""
        return StandardItem(
            id=c.id,
            text=f"{c.name}  ({c.image})",
            subtext=f"{prefix}{c.status}",
            icon_factory=lambda: self._container_icon(c),
            actions=actions,
            input_action_text=f"@{c.project} " if c.project else c.name,
        )

    def _image_items(self, query):
        matcher = Matcher(query)
        for image in reversed(self.client.images.list()):
            if match := (matcher.match(image.tags) if query else True):
                yield RankItem(
                    StandardItem(
                        id=image.short_id,
                        text=", ".join(image.tags) or image.short_id,
                        subtext="Image: %s" % image.id,
                        icon_factory=lambda: Icon.composed(Icon.image(self.icon_blue),
                                                           Icon.grapheme("💿")),
                        actions=[Action("rmi", "Remove image", lambda i=image: i.remove())]
                    ),
                    match if query else 1.0
                )

    def rankItems(self, ctx):
        if not self.client:
            try:
                self.client = docker.from_env()
            except Exception as e:
                return [RankItem(StandardItem(
                    id='except',
                    text="Failed starting docker client",
                    subtext=str(e),
                    icon_factory=lambda: self._icon(True, "⚠️")
                ), 0)]

        query = ctx.query.strip()
        try:
            # Images
            if query == "img" or query.startswith("img "):
                return list(self._image_items(query[3:].strip()))

            projects, containers = self._fetch()

            # Containers of one project: "@project [filter]"
            if query.startswith("@"):
                name, _, rest = query[1:].partition(" ")
                if name in projects and (rest or ctx.query.endswith(" ")):
                    matcher = Matcher(rest.strip())
                    project = projects[name]
                    items = [RankItem(self._project_item(project), 1.0)] if not rest.strip() else []
                    for c in sorted(project.containers, key=lambda c: c.name):
                        if not rest.strip():
                            # running first, then alphabetic (sort above is stable)
                            items.append(RankItem(self._container_item(c, projects),
                                                  0.9 if c.running else 0.5))
                        elif m := matcher.match(c.name, c.service or ""):
                            items.append(RankItem(self._container_item(c, projects), m))
                    return items
                # Still typing the project name: complete it
                matcher = Matcher(name)
                return [RankItem(self._project_item(project), m)
                        for project in projects.values() if (m := matcher.match(project.name))]

            # Empty query: projects (running first) and containers outside any project
            if not query:
                items = [RankItem(self._project_item(project), 0.9 if project.running else 0.4)
                         for project in projects.values()]
                items += [RankItem(self._container_item(c, projects),
                                   0.8 if c.running else 0.3)
                          for c in containers if not c.project]
                return items

            # Search: projects by name, containers by name and service
            matcher = Matcher(query)
            items = []
            if any(c.running for c in containers) and (m := matcher.match("stop all")):
                items.append(RankItem(self._stop_all_item(containers), m))
            items += [RankItem(self._project_item(project), m)
                     for project in projects.values() if (m := matcher.match(project.name))]
            items += [RankItem(self._container_item(c, projects), m)
                      for c in containers
                      if (m := matcher.match(c.name, c.service or ""))]
            return items

        except Exception as e:
            warning(str(e))
            self.client = None
            self._invalidate()
            return []
