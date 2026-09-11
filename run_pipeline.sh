#!/bin/bash
# ==============================================================================
# Dual Arm Imitation Learning: Full Pipeline Script 
# ==============================================================================

set -e  # 에러 발생 시 즉시 종료

# 설정 (자율적으로 수정 가능)
ALGO="diffusion"
NUM_DEMOS=2000   # Handover는 난이도가 높으므로 500개 권장
EPOCHS=1500     # 학습 에폭 (150은 너무 적으므로 1500 추천)
RUN_SEQ_EVAL=false  # 순차 평가(eval.py)를 실행할지 여부 (true/false)
PYTHON_EXEC="~/isaac_lab/bin/python"

echo "============================================================================="
echo "🚀 파이프라인 시작 (Data Collection -> Train -> Eval)"
echo "알고리즘: $ALGO | 데모 개수: $NUM_DEMOS | 학습 에폭: $EPOCHS"
echo "============================================================================="

# 1. 기존 결과 파일 초기화
> pipeline_summary.txt

# 2. 데이터 수집
echo -e "\n[1/4] 🎥 $NUM_DEMOS 개의 데모 데이터 수집 중..."
eval $PYTHON_EXEC scripts/generate_scripted_demos_parallel.py --num_demos $NUM_DEMOS --num_envs 16 --headless
echo "✅ 데이터 수집 완료."

# 3. 모델 학습
echo -e "\n[2/4] 🧠 $ALGO 모델 학습 중 ($EPOCHS Epochs)..."
eval $PYTHON_EXEC scripts/train.py --algo $ALGO --epochs $EPOCHS
echo "✅ 학습 완료."

# 4. 순차 평가 (eval.py)
if [ "$RUN_SEQ_EVAL" = true ] || [ "$RUN_SEQ_EVAL" = "true" ]; then
    echo -e "\n[3/4] ⚖️ 순차 평가 (eval.py) 10 에피소드 진행 중..."
    SEQ_LOG="results_eval_seq.txt"
    eval $PYTHON_EXEC scripts/eval.py --algo $ALGO --num_episodes 10 --headless > $SEQ_LOG 2>&1 || true
    SUCCESS_COUNT=$(grep -c "SUCCESS!" $SEQ_LOG || true)
    echo "✅ 순차 평가 완료. (성공 횟수: $SUCCESS_COUNT / 10)"
    echo "[eval.py] Success Count: $SUCCESS_COUNT / 10" >> pipeline_summary.txt
else
    echo -e "\n[3/4] ⏭️ 순차 평가 (eval.py) 건너뜀 (RUN_SEQ_EVAL=false)"
fi

# 5. 병렬 평가 (eval_parallel.py)
echo -e "\n[4/4] 🚀 병렬 평가 (eval_parallel.py) 100 에피소드 진행 중..."
PAR_LOG="results_eval_parallel.txt"
eval $PYTHON_EXEC scripts/eval_parallel.py --algo $ALGO --num_episodes 100 --num_envs 16 --headless > $PAR_LOG 2>&1 || true
PAR_SUCCESS=$(grep "Success Rate" $PAR_LOG | tail -n 1 | xargs)
echo "✅ 병렬 평가 완료. ($PAR_SUCCESS)"
echo "[eval_parallel.py] $PAR_SUCCESS" >> pipeline_summary.txt

echo "============================================================================="
echo "🎉 모든 파이프라인이 종료되었습니다! 최종 결과 요약:"
cat pipeline_summary.txt
echo "============================================================================="
