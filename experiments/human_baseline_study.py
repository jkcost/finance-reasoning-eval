"""Human Baseline Study: IC (Information Conflict) Detection.

IC-L3 문제 선별, HTML 설문 생성, 인간 vs LLM 비교 채점.

Usage:
    python experiments/human_baseline_study.py
    python experiments/human_baseline_study.py --score --human-results r.json --llm-results l.json
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results" / "metacognitive"
DEFAULT_TRANSFORMATIONS = RESULTS_DIR / "ic_ladder_transformations_0_120.json"
DEFAULT_SURVEY_OUTPUT = RESULTS_DIR / "human_baseline_survey.html"


def _classify_context_type(context: str) -> str:
    """컨텍스트 형식을 'json', 'markdown', 'text' 중 하나로 분류한다."""
    stripped = context.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return "json"
    if "|" in context[:500] and "---" in context[:500]:
        return "markdown"
    return "text"


def select_baseline_problems(
    transformations_path: Path,
    n: int = 15,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """IC-L3 변환 문제 중 대표 문제를 선별한다 (유형 혼합, 중복 제거).

    Args:
        transformations_path: IC ladder 변환 JSON 경로.
        n: 선별할 문제 수.
        seed: 재현 가능한 랜덤 시드.
    """
    with open(transformations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_transforms = data.get("transformations", [])
    l3_transforms = [
        t for t in all_transforms if "L3" in t.get("transformation_type", "")
    ]
    logger.info("IC-L3 변환 수: %d / 전체 %d", len(l3_transforms), len(all_transforms))

    if not l3_transforms:
        logger.warning("IC-L3 변환이 없습니다. 전체 변환에서 선별합니다.")
        l3_transforms = all_transforms

    # question_id별 첫 번째 변환만 사용 (중복 제거)
    seen_ids: set = set()
    unique_transforms: List[Dict[str, Any]] = []
    for t in l3_transforms:
        qid = t["question_id"]
        if qid not in seen_ids:
            seen_ids.add(qid)
            t["context_type"] = _classify_context_type(t["context"])
            unique_transforms.append(t)

    logger.info("고유 문제 수: %d", len(unique_transforms))

    # 컨텍스트 유형별 그룹핑
    by_type: Dict[str, List[Dict]] = {"json": [], "markdown": [], "text": []}
    for t in unique_transforms:
        by_type[t["context_type"]].append(t)

    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []

    # 유형별 최소 할당: 있는 유형에서 최소 1개씩
    for ctx_type, items in by_type.items():
        if items and len(selected) < n:
            pick = rng.sample(items, min(len(items), max(1, n // 5)))
            selected.extend(pick)

    # 나머지는 전체 풀에서 채움
    remaining_ids = {t["question_id"] for t in selected}
    pool = [t for t in unique_transforms if t["question_id"] not in remaining_ids]
    rng.shuffle(pool)
    selected.extend(pool[: max(0, n - len(selected))])

    selected = selected[:n]
    rng.shuffle(selected)

    type_counts = {
        ct: sum(1 for t in selected if t["context_type"] == ct)
        for ct in {"json", "markdown", "text"}
        if any(t["context_type"] == ct for t in selected)
    }
    logger.info("선별 결과: %d문제 (유형: %s)", len(selected), type_counts)

    return selected


def generate_baseline_survey(
    problems: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """자체 완결형 HTML 설문을 생성한다 (타이머, 신뢰도, JSON 내보내기 포함).

    Args:
        problems: select_baseline_problems에서 반환된 문제 리스트.
        output_path: 생성할 HTML 파일 경로.
    """
    problems_json = json.dumps(
        [
            {
                "id": p["question_id"],
                "question": p["question"],
                "context": p["context"],
                "context_type": p.get("context_type", "text"),
                "transformation_type": p.get("transformation_type", ""),
                "transformation_description": p.get("transformation_description", ""),
            }
            for p in problems
        ],
        ensure_ascii=False,
    )

    html_content = _build_survey_html(problems_json, len(problems))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info("설문 HTML 생성 완료: %s", output_path)


def _build_survey_html(problems_json: str, problem_count: int) -> str:
    """설문 HTML 전체를 구성한다."""
    # CSS와 JS를 최소화하여 인라인 포함
    css = (
        "*{margin:0;padding:0;box-sizing:border-box}"
        "body{font-family:'Segoe UI',sans-serif;background:#f8f9fa;color:#333;line-height:1.6}"
        ".c{max-width:900px;margin:0 auto;padding:20px}"
        "h1{text-align:center;margin:20px 0;color:#1a1a2e}"
        ".intro{background:#fff;padding:20px;border-radius:8px;margin-bottom:20px;border-left:4px solid #3b82f6}"
        ".pi{background:#fff;padding:20px;border-radius:8px;margin-bottom:20px}"
        ".pi label{display:block;margin:8px 0 4px;font-weight:600}"
        ".pi input{width:100%;padding:8px;border:1px solid #ddd;border-radius:4px}"
        ".pc{background:#fff;padding:20px;border-radius:8px;margin-bottom:16px;border:1px solid #e0e0e0}"
        ".ph{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}"
        ".pn{font-weight:700;font-size:1.1em;color:#1a1a2e}"
        ".tm{font-size:.9em;color:#666;font-family:monospace}"
        ".cb{background:#f1f5f9;padding:12px;border-radius:6px;margin:8px 0;white-space:pre-wrap;"
        "font-family:Consolas,monospace;font-size:.85em;max-height:400px;overflow-y:auto;border:1px solid #e2e8f0}"
        ".qt{font-weight:600;margin:12px 0 8px;padding:8px;background:#eff6ff;border-radius:4px}"
        ".cg{margin:10px 0}.cg label{display:block;padding:6px 8px;margin:4px 0;border-radius:4px;cursor:pointer}"
        ".cg label:hover{background:#f0f0f0}.cg input[type=radio]{margin-right:8px}"
        ".af,.if{margin:8px 0}.af textarea,.if textarea{width:100%;min-height:60px;padding:8px;"
        "border:1px solid #ddd;border-radius:4px;font-family:inherit}"
        ".cf{margin:10px 0;display:flex;align-items:center;gap:10px}"
        ".cf input[type=range]{flex:1}.cv{font-weight:700;min-width:20px;text-align:center}"
        ".h{display:none}.act{text-align:center;margin:20px 0}"
        ".btn{padding:10px 24px;border:none;border-radius:6px;font-size:1em;cursor:pointer;margin:4px}"
        ".bp{background:#3b82f6;color:#fff}.bp:hover{background:#2563eb}"
        ".bs{background:#10b981;color:#fff}.bs:hover{background:#059669}"
        ".pg{background:#e0e0e0;border-radius:10px;height:8px;margin:10px 0}"
        ".pb{background:#3b82f6;height:100%;border-radius:10px;transition:width .3s}"
        ".sm{background:#fff;padding:20px;border-radius:8px;margin-top:20px;border:2px solid #10b981}"
    )
    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>Human Baseline Study: IC Detection</title>
<style>{css}</style></head><body><div class="c">
<h1>Human Baseline Study: Information Conflict Detection</h1>
<div class="intro">
<p><b>목적:</b> 금융 데이터에 삽입된 정보 충돌(모순)을 인간이 얼마나 잘 탐지하는지 측정합니다.</p>
<p><b>지시사항:</b> 각 문제의 컨텍스트와 질문을 읽고 아래 중 하나를 선택하세요:</p>
<ul style="margin:8px 0 0 20px"><li><b>풀 수 있음</b> - 답을 작성</li>
<li><b>풀 수 없음</b> - 정보 부족</li>
<li><b>데이터에 문제가 있음</b> - 모순/충돌 발견 시 설명</li></ul>
<p style="margin-top:8px">총 <b>{problem_count}문제</b>, 문제당 소요시간 자동 측정</p></div>
<div class="pi" id="ps"><h3>참가자 정보</h3>
<label for="pid">참가자 ID</label><input type="text" id="pid" placeholder="예: participant_01">
<label for="exp">금융 경력 (년)</label><input type="number" id="exp" min="0" max="50" value="0">
<div style="margin-top:12px"><button class="btn bp" onclick="startSurvey()">시작</button></div></div>
<div id="sb" class="h"><div class="pg"><div class="pb" id="pb"></div></div>
<div id="pc"></div><div class="act">
<button class="btn bs" onclick="exportResults()">결과 내보내기 (JSON)</button></div></div>
<div id="ss" class="h"></div></div>
<script>
const P={problems_json};
let ST={{}},sst=null;
function $(id){{return document.getElementById(id)}}
function escH(s){{let d=document.createElement('div');d.textContent=s;return d.innerHTML}}
function startSurvey(){{
  if(!$('pid').value.trim()){{alert('참가자 ID를 입력해주세요.');return}}
  $('ps').classList.add('h');$('sb').classList.remove('h');sst=Date.now();render()}}
function render(){{
  let c=$('pc');c.innerHTML='';
  P.forEach((p,i)=>{{
    let d=document.createElement('div');d.className='pc';d.id='p'+i;
    d.innerHTML=`<div class="ph"><span class="pn">문제 ${{i+1}}/${{P.length}}</span>
      <span class="tm" id="t${{i}}">00:00</span></div>
      <div class="cb">${{escH(p.context)}}</div><div class="qt">${{escH(p.question)}}</div>
      <div class="cg">
        <label><input type="radio" name="c${{i}}" value="solvable" onchange="oc(${{i}},this.value)"> 풀 수 있음</label>
        <label><input type="radio" name="c${{i}}" value="unsolvable" onchange="oc(${{i}},this.value)"> 풀 수 없음</label>
        <label><input type="radio" name="c${{i}}" value="conflict_detected" onchange="oc(${{i}},this.value)"> 데이터에 문제가 있음</label>
      </div>
      <div class="af h" id="a${{i}}"><label><b>답변:</b></label><textarea id="at${{i}}"></textarea></div>
      <div class="if h" id="is${{i}}"><label><b>발견한 문제:</b></label><textarea id="it${{i}}" placeholder="모순 설명"></textarea></div>
      <div class="cf"><span>신뢰도:</span><span>1</span>
        <input type="range" min="1" max="5" value="3" id="cf${{i}}" oninput="$('cv${{i}}').textContent=this.value">
        <span>5</span><span class="cv" id="cv${{i}}">3</span></div>`;
    c.appendChild(d);ST[i]=Date.now()}});tick()}}
function oc(i,v){{$('a'+i).classList.toggle('h',v!=='solvable');
  $('is'+i).classList.toggle('h',v!=='conflict_detected');up()}}
function up(){{let n=0;P.forEach((_,i)=>{{if(document.querySelector(`input[name="c${{i}}"]:checked`))n++}});
  $('pb').style.width=(n/P.length*100)+'%'}}
function tick(){{P.forEach((_,i)=>{{let e=$('t'+i);if(e&&ST[i]){{let s=Math.floor((Date.now()-ST[i])/1000);
  e.textContent=String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0')}}}});
  requestAnimationFrame(tick)}}
function collect(){{let r={{participant_id:$('pid').value.trim(),experience_years:+$('exp').value||0,
  survey_start:new Date(sst).toISOString(),survey_end:new Date().toISOString(),responses:[]}};
  P.forEach((p,i)=>{{let ck=document.querySelector(`input[name="c${{i}}"]:checked`);
    r.responses.push({{question_id:p.id,transformation_type:p.transformation_type,
      choice:ck?ck.value:null,answer:$('at'+i).value.trim()||null,
      issue_description:$('it'+i).value.trim()||null,
      confidence:+$('cf'+i).value,elapsed_seconds:Math.round((Date.now()-ST[i])/1000)}})}});return r}}
function exportResults(){{let r=collect(),u=r.responses.filter(x=>!x.choice).length;
  if(u>0&&!confirm(u+'개 미응답. 계속?'))return;
  let b=new Blob([JSON.stringify(r,null,2)],{{type:'application/json'}}),a=document.createElement('a');
  a.href=URL.createObjectURL(b);a.download='human_baseline_'+r.participant_id+'.json';a.click();
  let t=r.responses.length,ans=r.responses.filter(x=>x.choice).length,
    cd=r.responses.filter(x=>x.choice==='conflict_detected').length,
    avg=Math.round(r.responses.reduce((s,x)=>s+x.elapsed_seconds,0)/t);
  $('ss').innerHTML=`<div class="sm"><h3>제출 완료</h3><p>응답: ${{ans}}/${{t}}</p>
    <p>충돌 탐지: ${{cd}}건</p><p>평균: ${{avg}}초/문제</p></div>`;$('ss').classList.remove('h')}}
</script></body></html>"""


def score_baseline_results(
    human_results_path: Path,
    llm_results_path: Path,
) -> Dict[str, Any]:
    """인간 응답과 LLM 응답을 비교 채점한다 (conflict_detected = 정답).

    Args:
        human_results_path: 설문 JSON 경로 (파일 또는 디렉토리).
        llm_results_path: LLM 배치 평가 결과 JSON 경로.
    """
    # 인간 결과 로드 (단일 파일 또는 디렉토리)
    human_responses = _load_human_results(human_results_path)
    llm_responses = _load_llm_results(llm_results_path)

    # 인간 집계
    human_total = len(human_responses)
    human_detected = sum(
        1 for r in human_responses if r.get("choice") == "conflict_detected"
    )
    human_solvable = sum(1 for r in human_responses if r.get("choice") == "solvable")
    human_unsolvable = sum(
        1 for r in human_responses if r.get("choice") == "unsolvable"
    )

    # LLM 집계 (IC 문제에서 refused = 충돌 감지로 간주)
    llm_total = len(llm_responses)
    llm_refused = sum(1 for r in llm_responses if r.get("response_type") == "refused")

    # 문제별 일치율 (같은 question_id에 대해)
    human_by_qid: Dict[str, str] = {}
    for r in human_responses:
        qid = r["question_id"]
        human_by_qid[qid] = r.get("choice", "none")

    llm_by_qid: Dict[str, str] = {}
    for r in llm_responses:
        qid = r["question_id"]
        is_detected = r.get("response_type") == "refused"
        llm_by_qid[qid] = "conflict_detected" if is_detected else "solvable"

    common_qids = set(human_by_qid.keys()) & set(llm_by_qid.keys())
    agreements = sum(1 for qid in common_qids if human_by_qid[qid] == llm_by_qid[qid])

    details = [
        {
            "question_id": qid,
            "human_choice": human_by_qid[qid],
            "llm_choice": llm_by_qid[qid],
            "agreed": human_by_qid[qid] == llm_by_qid[qid],
        }
        for qid in sorted(common_qids)
    ]

    result = {
        "human_stats": {
            "total": human_total,
            "conflict_detected": human_detected,
            "solvable": human_solvable,
            "unsolvable": human_unsolvable,
            "detection_rate": human_detected / human_total if human_total else 0,
        },
        "llm_stats": {
            "total": llm_total,
            "refused": llm_refused,
            "detection_rate": llm_refused / llm_total if llm_total else 0,
        },
        "comparison": {
            "common_questions": len(common_qids),
            "agreements": agreements,
            "agreement_rate": agreements / len(common_qids) if common_qids else 0,
        },
        "human_detection_details": details,
    }

    logger.info(
        "채점 완료: 인간 탐지율=%.1f%%, LLM 탐지율=%.1f%%, 일치율=%.1f%%",
        result["human_stats"]["detection_rate"] * 100,
        result["llm_stats"]["detection_rate"] * 100,
        result["comparison"]["agreement_rate"] * 100,
    )
    return result


def _load_human_results(path: Path) -> List[Dict[str, Any]]:
    """인간 결과를 로드한다. 단일 파일 또는 디렉토리 지원."""
    responses: List[Dict[str, Any]] = []
    if path.is_dir():
        files = sorted(path.glob("human_baseline_*.json"))
        logger.info("인간 결과 파일 %d개 발견: %s", len(files), path)
        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for r in data.get("responses", []):
                r["participant_id"] = data.get("participant_id", f.stem)
                responses.append(r)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data.get("responses", []):
            r["participant_id"] = data.get("participant_id", "unknown")
            responses.append(r)
    logger.info("인간 응답 %d건 로드", len(responses))
    return responses


def _load_llm_results(path: Path) -> List[Dict[str, Any]]:
    """LLM 평가 결과를 로드하고 IC 문제만 필터링한다."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    ic_results = [r for r in results if "IC" in r.get("transformation_type", "")]
    logger.info("LLM IC 결과 %d건 로드 (전체 %d건)", len(ic_results), len(results))
    return ic_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Human Baseline Study: IC Detection")
    parser.add_argument("--score", action="store_true", help="채점 모드 실행")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_TRANSFORMATIONS,
        help="IC 변환 JSON 경로",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SURVEY_OUTPUT,
        help="설문 HTML 출력 경로",
    )
    parser.add_argument("--n", type=int, default=15, help="선별 문제 수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--human-results", type=Path, help="인간 결과 JSON 경로")
    parser.add_argument("--llm-results", type=Path, help="LLM 결과 JSON 경로")

    args = parser.parse_args()

    if args.score:
        if not args.human_results or not args.llm_results:
            parser.error(
                "--score 모드에는 --human-results와 --llm-results가 필요합니다"
            )
        result = score_baseline_results(args.human_results, args.llm_results)
        output = args.output.with_suffix(".json")
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("채점 결과 저장: %s", output)
    else:
        problems = select_baseline_problems(args.input, n=args.n, seed=args.seed)
        generate_baseline_survey(problems, args.output)


if __name__ == "__main__":
    main()
