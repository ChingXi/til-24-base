import asyncio
import json
import logging
import os
from time import time
from typing import Any, Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

#From nlp_eval.py:
from statistics import mean
from rouge_score import rouge_scorer
from typing import List, Dict

ROUGEL = "rougeL"
scorer = rouge_scorer.RougeScorer([ROUGEL], use_stemmer=True)

keys = ("heading", "target", "tool")


def score(key: str, ref: str, hyp: str) -> float:
    # exact match for heading
    if key == "heading":
        return 1.0 if ref == hyp else 0.0
    # ROUGE-L for everything else
    else:
        score = scorer.score(ref, hyp)[ROUGEL]
        return score.fmeasure


def nlp_eval(truth: List[Dict[str, str]], hypothesis: List[Dict[str, str]]) -> float:
    results = []
    for ref, hyp in zip(truth, hypothesis):
        results.append(mean(score(key, ref[key], hyp[key]) for key in keys))
    return mean(results)




#From vlm_eval.py:
from statistics import mean
from typing import List, Optional


def bb_iou(bb1: Optional[List[int]], bb2: List[int]) -> int:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes in ltwh format.

    Parameters
    ----------
    bb1 : list[int, int, int, int] | None
        left, top, width, height
    bb2 : list[int, int, int, int]
        left, top, width, height

    Returns
    -------
    int
        0 or 1
    """
    if not bb1:
        return 0
    boxA = [bb1[0], bb1[1], bb1[0] + bb1[2], bb1[1] + bb1[3]]
    boxB = [bb2[0], bb2[1], bb2[0] + bb2[2], bb2[1] + bb2[3]]

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0.0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value @ 0.5
    return round(iou)


def vlm_eval(
    bbox_truths: List[Optional[List[int]]], bbox_predictions: List[List[int]]
) -> float:
    return mean(
        bb_iou(bb_truth, bb_pred)
        for bb_truth, bb_pred in zip(bbox_truths, bbox_predictions)
    )



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

prefix: str = "results/"
TEAM_NAME: str = os.getenv("TEAM_NAME", "team-name")

# filepath to load testcase data from
filepath = "simulator/data/data.json"
with open(filepath) as f:
    testcases = json.load(f)
os.makedirs(prefix, exist_ok=True)

app = FastAPI()
app.mount("/simulator", StaticFiles(directory="simulator"), name="simulator")


# serve simulator html
@app.get("/")
async def read_index():
    return FileResponse("simulator/index.html")


@app.get("/health")
def health():
    return {"health": "ok"}


# websocket connection manager
class MockConnectionManager:
    def __init__(self):
        self.simulator_connection: WebSocket | None = None
        self.team_connection: WebSocket | None = None
        self.team_bbox: List[int] = [0, 0, 0, 0]
        self.autonomy_connection: WebSocket | None = None

    async def simulator_connect(self, websocket: WebSocket):
        if self.simulator_connection is not None:
            logger.info("disconnecting previous connection")
            try:
                await self.simulator_connection.close()
            except Exception as e:
                logger.exception(e)
            self.simulator_connection = None
        await websocket.accept()
        self.simulator_connection = websocket
        await websocket.send_json(
            {
                "type": "teams",
                "teams": [TEAM_NAME],
            }
        )

    async def team_connect(self, websocket: WebSocket):
        if self.team_connection == None:
            await websocket.accept()
            self.team_connection = websocket
        else:
            # await websocket.close(
            #     reason=f"There is already a team connected with id {team_id}!"
            # )
            await self.team_connection.close()
            await websocket.accept()
            self.team_connection = websocket

    async def autonomy_connect(self, websocket: WebSocket):
        if self.autonomy_connection == None:
            await websocket.accept()
            self.autonomy_connection = websocket
        else:
            # await websocket.close(
            #     reason=f"There is already a team connected with id {team_id}!"
            # )
            await self.autonomy_connection.close()
            await websocket.accept()
            self.autonomy_connection = websocket

    async def update_simulator(self, websocket: WebSocket, team_name: int, data: dict):
        if self.simulator_connection is not None:
            await self.simulator_connection.send_json({"name": team_name, **data})
            if data["type"] == "snapshot":
                bbox = await self.simulator_connection.receive_json()
                snapshot = await self.simulator_connection.receive_bytes()
                await websocket.send_bytes(snapshot)
                if bbox["valid"]:
                    self.team_bbox = [
                        bbox[side] for side in ("left", "top", "width", "height")
                    ]
                else:
                    self.team_bbox = None
                logger.info("sending snapshot and writing file")
                with open(f"{prefix}team_{team_name}_snapshot.jpg", "wb") as file:
                    file.write(snapshot)

    def disconnect(self, team_name: str):
        self.team_connection = None
        self.autonomy_connection = None


manager = MockConnectionManager()


# websocket connection for simulator
@app.websocket("/ws_sim")
async def simulator_websocket(websocket: WebSocket):
    await manager.simulator_connect(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info("disconnected")
    finally:
        manager.simulator_disconnect()


@app.websocket("/ws/{team_name}")
async def team_endpoint(websocket: WebSocket, team_name: str):
    await manager.team_connect(websocket)
    try:
        responses: List[Dict[str, Any]] = []
        for case in testcases:
            with open("simulator/data/audio/" + case["audio"], "rb") as file:
                audio_bytes = file.read()
            # determine time between start and end
            start_time = time()
            await websocket.send_bytes(audio_bytes)
            results = await websocket.receive_json()
            # evaluate responses
            elapsed = time() - start_time
            logger.info(f"Team {team_name} took {elapsed:.3f}s")
            results["bbox"] = manager.team_bbox
            results["truth"] = case["truth"]
            results["elapsed"] = elapsed
            results["nlp_score"] = nlp_eval([case["truth"]], [results["nlp"]])
            results["vlm_score"] = vlm_eval([results["bbox"]], [results["vlm"]])
            results["perf_score"] = 1 - min(30, elapsed) / 30
            results["score"] = (
                0.45 * results["nlp_score"]
                + 0.45 * results["vlm_score"]
                + 0.1 * results["perf_score"]
            )
            logger.info(results)
            responses.append(results)
            await manager.simulator_connection.send_json({"type": "switch"})
        await websocket.send_json({"status": "done"})
        # write responses
        with open(prefix + f"team_{team_name}_results.jsonl", "w") as f:
            for response in responses:
                f.write(json.dumps(response) + "\n")
    except WebSocketDisconnect:
        logger.info(f"Team {team_name} disconnected")
    finally:
        manager.disconnect(team_name)


@app.websocket("/ws_auto/{team_name}")
async def autonomy_endpoint(websocket: WebSocket, team_name: str):
    await manager.autonomy_connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            await manager.update_simulator(websocket, team_name, data)
    except WebSocketDisconnect:
        manager.disconnect(team_name)
