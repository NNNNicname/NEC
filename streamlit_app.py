"""
NEC 手术风险在线预测系统
基于 CatBoost + SHAP 可解释性分析
运行方式: streamlit run streamlit_app.py
"""
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import catboost as cb
import shap
import matplotlib.pyplot as plt

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="NEC 手术风险预测",
    page_icon="🏥",
    layout="wide",
)

# ==================== 中文绘图支持 ====================
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 文件路径（模型文件与此脚本同目录） ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "catboost_model.cbm")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "catboost_feature_names.pkl")
SHAP_PATH = os.path.join(BASE_DIR, "catboost_shap_result.pkl")


# ==================== 加载模型与数据（缓存） ====================
@st.cache_resource
def load_model_and_data():
    """加载 CatBoost 模型、特征名和 SHAP 结果"""
    try:
        model = cb.CatBoostClassifier()
        model.load_model(MODEL_PATH)

        with open(FEATURE_NAMES_PATH, "rb") as f:
            feature_names = pickle.load(f)

        with open(SHAP_PATH, "rb") as f:
            shap_result = pickle.load(f)

        return model, feature_names, shap_result
    except FileNotFoundError as e:
        st.error(f"❌ 模型文件缺失：{e.filename}")
        return None, None, None
    except Exception as e:
        st.error(f"❌ 模型加载失败：{e}")
        return None, None, None


model, feature_names, shap_result = load_model_and_data()

# ==================== 页面 UI ====================
st.title("🏥 坏死性小肠结肠炎（NEC）手术风险预测模型")
st.markdown("#### 基于 CatBoost 机器学习 + SHAP 可解释性分析")
st.markdown("---")

# ---------- 侧边栏：特征输入 ----------
with st.sidebar:
    st.header("📝 输入患者特征")

    # 特征中文名映射（可自行增改）
    cn_labels = {
        "CRP": "C 反应蛋白 CRP (mg/L)",
        "RBC": "红细胞计数 RBC (×10¹²/L)",
        "sbp": "收缩压 SBP (mmHg)",
        "dbp": "舒张压 DBP (mmHg)",
        "NLR": "中性粒/淋巴比值 NLR",
        "Lymph_Count": "淋巴细胞计数 (×10⁹/L)",
        "PLR": "血小板/淋巴比值 PLR",
        "Preterm_baby": "是否早产儿",
        "asphyxia": "是否窒息史",
        "sepsis": "是否败血症",
    }

    input_data = {}
    for feat in feature_names:
        label = cn_labels.get(feat, feat)
        if feat in ("Preterm_baby", "asphyxia", "sepsis"):
            input_data[feat] = st.selectbox(label, options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
        else:
            input_data[feat] = st.number_input(label, value=0.0, step=0.01, format="%.2f")

    st.markdown("---")
    predict_btn = st.button("🚀 开始预测", type="primary", use_container_width=True)

# ---------- 构造输入 DataFrame ----------
input_df = pd.DataFrame([input_data])

# ==================== 预测逻辑 ====================
if predict_btn and model is not None:
    # 1. 模型预测
    pred_prob = model.predict_proba(input_df)[0]   # [P(0类), P(1类)]
    pred_label = model.predict(input_df)[0]         # 0 或 1

    st.subheader("🔍 预测结果")
    col1, col2 = st.columns(2)
    with col1:
        risk_text = "⚠️ 高风险（建议手术）" if pred_label == 1 else "✅ 低风险（保守治疗）"
        st.metric("预测类别", risk_text)
    with col2:
        st.metric("手术风险概率", f"{pred_prob[1]:.2%}")

    # 2. 风险进度条
    st.progress(float(pred_prob[1]), text=f"风险评分：{pred_prob[1]:.1%}")

    st.markdown("---")

    # ==================== SHAP 可解释性 ====================
    st.subheader("📈 SHAP 特征贡献分析")

    explainer = shap.TreeExplainer(model)
    shap_values_single = explainer.shap_values(input_df)

    # -- 单样本瀑布图 (更直观) --
    st.markdown("#### 🔬 该患者的特征贡献（瀑布图）")
    fig_wf, ax_wf = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values_single[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0].values,
            feature_names=feature_names,
        ),
        show=False,
    )
    st.pyplot(fig_wf)

    # -- 单样本力图 --
    st.markdown("#### 📊 该患者的 SHAP 力图")
    fig_force, ax_force = plt.subplots(figsize=(14, 3))
    shap.plots.force(
        base_value=explainer.expected_value,
        shap_values=shap_values_single,
        features=input_df.iloc[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    st.pyplot(fig_force)

    # -- 全局特征重要性 --
    st.markdown("---")
    st.markdown("#### 🌐 全局特征重要性（基于全体样本 SHAP 均值）")

    fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
    shap.summary_plot(
        shap_result["shap_values"],
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    st.pyplot(fig_bar)

    # -- SHAP 摘要图 --
    st.markdown("#### 📋 SHAP 摘要图（全体样本）")
    fig_summary, ax_summary = plt.subplots(figsize=(8, 5))
    shap.summary_plot(
        shap_result["shap_values"],
        feature_names=feature_names,
        show=False,
    )
    st.pyplot(fig_summary)

    # -- 单特征依赖图 --
    st.markdown("---")
    st.markdown("#### 🔎 单特征依赖图（可选）")
    selected_feat = st.selectbox("选择要查看的特征", feature_names, key="dep_feat")
    fig_dep, ax_dep = plt.subplots(figsize=(7, 4))
    shap.dependence_plot(
        selected_feat,
        shap_result["shap_values"],
        pd.DataFrame(
            shap_result["shap_values"],
            columns=feature_names,
        ),
        feature_names=feature_names,
        show=False,
    )
    st.pyplot(fig_dep)

elif predict_btn and model is None:
    st.error("模型未成功加载，无法预测。请检查模型文件是否完整。")

# ==================== 页脚 ====================
st.markdown("---")
st.markdown(
    "💡 **免责声明**：本模型仅供临床辅助参考，最终诊疗决策请结合医生专业判断。"
)
