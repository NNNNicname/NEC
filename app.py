

import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
try:
    import catboost as cb
except ImportError:
    install("catboost==1.4.0")
    import catboost as cb

# 导入需要的库
import streamlit as st
import pandas as pd
import numpy as np
import catboost as cb
import pickle
import shap
import matplotlib.pyplot as plt


# ========== 基础配置（解决中文显示问题） ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False    # 显示负号
st.set_page_config(page_title="NEC手术风险预测", layout="wide")  # 网页标题和布局

# ========== 加载模型和SHAP结果（缓存，避免重复加载） ==========
@st.cache_resource  # 缓存模型，只加载一次
def load_all_data():
    try:
        # 1. 加载CatBoost模型（原生.cbm格式）
        model = cb.CatBoostClassifier()  # 二分类模型，和你R端一致
        model.load_model("catboost_model.cbm")  # 加载你保存的模型
        
        # 2. 加载特征名
        with open("catboost_feature_names.pkl", 'rb') as f:
            feature_names = pickle.load(f)
        
        # 3. 加载SHAP结果
        with open("catboost_shap_result.pkl", 'rb') as f:
            shap_result = pickle.load(f)
        
        return model, feature_names, shap_result
    except Exception as e:
        st.error(f"模型加载失败！错误原因：{str(e)}")
        return None, None, None

# 调用加载函数
model, feature_names, shap_result = load_all_data()

# ========== 网页界面设计 ==========
st.title("📊 坏死性小肠结肠炎（NEC）手术风险预测模型")
st.markdown("### 基于CatBoost+SHAP可解释性分析")

# 侧边栏：特征输入（小白友好，每个特征都有输入框）
st.sidebar.header("📝 输入患者特征")
input_data = {}
for feat in feature_names:
    # 根据特征类型设置默认值（你可以根据实际数据调整）
    if feat in ["Preterm_baby", "asphyxia", "sepsis"]:  # 二分类特征（0/1）
        input_data[feat] = st.sidebar.selectbox(f"{feat}（0=否，1=是）", [0, 1])
    else:  # 连续型特征
        input_data[feat] = st.sidebar.number_input(f"{feat}（数值）", value=0.0, step=0.1)

# 转换为模型输入格式（DataFrame）
input_df = pd.DataFrame([input_data])

# ========== 预测和SHAP分析 ==========
if st.button("🚀 开始预测", type="primary"):
    if model is not None:
        # 1. 模型预测
        pred_prob = model.predict_proba(input_df)[0]  # 预测概率（0类/1类）
        pred_label = model.predict(input_df)[0]       # 预测类别
        
        # 显示预测结果（小白能看懂的格式）
        st.subheader("🔍 预测结果")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("预测类别", "需要手术" if pred_label == 1 else "无需手术")
        with col2:
            st.metric("手术风险概率", f"{pred_prob[1]:.2%}")  # 百分比显示
        
        # 2. SHAP可解释性分析（核心）
        st.subheader("📈 SHAP特征贡献分析")
        
        # 初始化SHAP解释器
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        # 2.1 单样本SHAP力图（Force Plot）
        st.markdown("#### 单个患者特征贡献（SHAP Force Plot）")
        shap_force = shap.force_plot(
            explainer.expected_value,
            shap_values,
            input_df,
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        st.pyplot(shap_force)
        
        # 2.2 全局特征重要性（SHAP Bar Plot）
        st.markdown("#### 全局特征重要性（SHAP Bar Plot）")
        fig, ax = plt.subplots()
        shap.summary_plot(
            shap_result['shap_values'],
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            ax=ax
        )
        st.pyplot(fig)
        
        # 2.3 特征依赖图（可选，小白可自选特征）
        st.markdown("#### 特征依赖图（单个特征对预测的影响）")
        selected_feat = st.selectbox("选择要分析的特征", feature_names)
        fig2, ax2 = plt.subplots()
        shap.dependence_plot(
            selected_feat,
            shap_result['shap_values'],
            pd.DataFrame(shap_result['shap_values'], columns=feature_names),
            feature_names=feature_names,
            show=False,
            ax=ax2
        )
        st.pyplot(fig2)
    else:
        st.error("❌ 模型加载失败！请检查模型文件是否存在，或联系技术人员。")

# 页脚（小白友好提示）
st.markdown("---")
st.markdown("💡 提示：本模型仅为临床辅助参考，最终决策请结合医生专业判断。")
