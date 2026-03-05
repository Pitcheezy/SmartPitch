import wandb
import os
import sys

# src 패키지를 인식할 수 있도록 프로젝트 루트 경로를 sys.path에 추가합니다.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 생성한 모듈 불러오기
from src.data_loader import PitchDataLoader
from src.clustering import PitchClustering
from src.model import TransitionProbabilityModel
from src.mdp_solver import MDPOptimizer

def main(player_first_name, player_last_name, start_date, end_date):
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    print("\n[System] 파이프라인 실행을 시작합니다...")
    print("=" * 60)
    print("SmartPitch 파이프라인 실행 시작")
    print("=" * 60)
    
    # -------------------------------------------------------------
    # 1. W&B 초기화 (전체 파이프라인 로깅용)
    # -------------------------------------------------------------
    run = wandb.init(
        project="SmartPitch-Portfolio",
        name=f"{player_first_name.capitalize()}_{player_last_name.capitalize()}_Pipeline",
        config={
            "pitcher": f"{player_first_name} {player_last_name}",
            "season": start_date[:4],
            "model_type": "PyTorch MLP",
            "epochs": 5,
            "hidden_dims": [128, 64],
            "dropout_rate": 0.2,
            "learning_rate": 0.001
        }
    )
    
    try:
        # -------------------------------------------------------------
        # 2. 데이터 수집 및 전처리 (PitchDataLoader)
        # -------------------------------------------------------------
        print("\n[단계 1] 투구 데이터 수집 및 전처리")
        data_loader = PitchDataLoader(
            first_name=player_first_name,
            last_name=player_last_name,
            start_date=start_date,
            end_date=end_date
        )
        
        # 데이터를 가져오고 전처리한 뒤, 바로 W&B Artifact로 업로드합니다.
        df_processed = data_loader.load_and_prepare_data(upload_artifact=True, artifact_name="gerrit_cole_raw_pitches")
        
        # -------------------------------------------------------------
        # 3. UMAP 및 K-Means 구종 동적 식별 (PitchClustering)
        # -------------------------------------------------------------
        print("\n[단계 2] UMAP 차원 축소 및 최적 군집(구종) 탐색")
        clustering = PitchClustering(df_processed)
        df_clustered = clustering.run_clustering_pipeline()
        
        # MDPOptimizer에 전달할 식별된 구종 목록 추출
        identified_pitch_names = list(clustering.pitch_map.values())
        print(f"식별된 구종 목록: {identified_pitch_names}")
        
        # -------------------------------------------------------------
        # 4. 상태 전이 확률 예측 딥러닝 모델 학습 (TransitionProbabilityModel)
        # -------------------------------------------------------------
        print("\n[단계 3] 상태 전이 확률 예측(MLP) 모델 학습")
        config = wandb.config
        model_module = TransitionProbabilityModel(
            df=df_clustered,
            batch_size=64,
            lr=config.learning_rate
        )
        
        # 데이터 분리, MLP 훈련, W&B 메트릭 로깅 및 가장 좋은 모델 가중치(Artifact) 업로드
        model_module.run_modeling_pipeline(
            epochs=config.epochs, 
            upload_artifact=True
        )
        
        # MDPOptimizer에 전달할 특징 컬럼과 타겟 클래스, 그리고 존(코스) 정보 추출
        feature_cols = model_module.feature_columns
        target_classes = model_module.target_classes
        strike_zones = sorted(list(df_clustered['zone'].dropna().unique()))
        
        # -------------------------------------------------------------
        # 5. 마르코프 결정 과정 (MDP) 기반 최적 정책 도출 (MDPOptimizer)
        # -------------------------------------------------------------
        print("\n[단계 4] 벨만 방정식 기반 최적 볼배합(Policy) 도출")
        optimizer = MDPOptimizer(
            transition_model=model_module,
            feature_columns=feature_cols,
            target_classes=target_classes,
            pitch_names=identified_pitch_names,
            zones=strike_zones
        )
        
        # 가치 반복 계산 및 W&B Table에 결과 로깅
        optimal_policy = optimizer.run_optimizer()
        
        print("\n" + "=" * 60)
        print("모든 SmartPitch 파이프라인이 성공적으로 완료되었습니다!")
        print("=" * 60)

    except Exception as e:
        print(f"\n파이프라인 실행 중 오류가 발생했습니다: {e}")
        raise e
        
    finally:
        # -------------------------------------------------------------
        # 6. W&B 세션 종료
        # -------------------------------------------------------------
        wandb.finish()

if __name__ == "__main__":
    print("=" * 60)
    print("⚾ SmartPitch AI: 메이저리그 투수 볼배합 최적화 파이프라인 ⚾")
    print("=" * 60)
    print("\n[데이터 수집 안내]")
    print("- 메이저리그 스탯캐스트(Statcast) 트래킹 데이터 기반입니다.")
    print("- 권장 데이터 기간: 2015년 ~ 2024년 정규시즌 (2015년 이전은 결측치가 많을 수 있습니다.)")
    print("\n[🎯 AI 성능 테스트를 위한 추천 투수 3인방]")
    print("1. Gerrit Cole   : 4구종(직구/슬/커/체)을 바탕으로 한 정석적인 투구 정책 베이스라인 테스트")
    print("2. Yu Darvish    : 10개 이상의 다양한 구종을 던지는 투수의 한계 군집화(Clustering) 테스트")
    print("3. Clayton Kershaw: 좌완 레전드의 예리한 슬라이더가 RE24 실점 억제에 미치는 영향 분석")
    print("\n" + "-" * 60)
    
    player_name = input("▶ 분석할 투수의 영문 이름을 입력하세요 (기본값: Gerrit Cole): ").strip()
    if not player_name:
        player_name = "Gerrit Cole"
        
    start_date = input("▶ 데이터 시작일을 입력하세요 (기본값: 2019-03-28): ").strip()
    if not start_date:
        start_date = "2019-03-28"
        
    end_date = input("▶ 데이터 종료일을 입력하세요 (기본값: 2019-09-29): ").strip()
    if not end_date:
        end_date = "2019-09-29"

    # 이름 분리 처리
    name_parts = player_name.split()
    if len(name_parts) >= 2:
        player_first_name = name_parts[0]
        player_last_name = " ".join(name_parts[1:])
    else:
        player_first_name = player_name
        player_last_name = ""

    main(player_first_name, player_last_name, start_date, end_date)
