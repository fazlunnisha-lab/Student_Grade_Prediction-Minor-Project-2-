import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 Student Grade Predictor")
st.markdown("Predict final grades based on assignment scores and attendance using Linear Regression.")

SAMPLE_DATA = pd.DataFrame({
    "Student": [f"S{i+1:03d}" for i in range(30)],
    "Assignment_1": np.round(np.random.RandomState(1).uniform(50, 100, 30), 1),
    "Assignment_2": np.round(np.random.RandomState(2).uniform(50, 100, 30), 1),
    "Assignment_3": np.round(np.random.RandomState(3).uniform(50, 100, 30), 1),
    "Midterm":      np.round(np.random.RandomState(4).uniform(45, 100, 30), 1),
    "Attendance":   np.round(np.random.RandomState(5).uniform(60, 100, 30), 1),
})
rng = np.random.RandomState(99)
SAMPLE_DATA["Final_Grade"] = np.round(
    0.15 * SAMPLE_DATA["Assignment_1"]
    + 0.15 * SAMPLE_DATA["Assignment_2"]
    + 0.15 * SAMPLE_DATA["Assignment_3"]
    + 0.30 * SAMPLE_DATA["Midterm"]
    + 0.25 * SAMPLE_DATA["Attendance"]
    + rng.normal(0, 3, 30),
    1,
).clip(0, 100)

with st.sidebar:
    st.header("🗂️ Dashboard")
    data_source = st.radio("Data Source", ["Use Sample Data", "Upload CSV", "Enter Manually"])
    st.markdown("---")
    st.subheader("Feature Selection")
    feature_options = ["Assignment_1", "Assignment_2", "Assignment_3", "Midterm", "Attendance"]
    selected_features = st.multiselect(
        "Predictors (X variables)",
        feature_options,
        default=feature_options,
    )
    test_size = st.slider("Test Split (%)", 10, 40, 20, step=5) / 100
    st.markdown("---")
    st.markdown("**Model:** Multiple Linear Regression")
    st.markdown("**Target:** Final Grade (0–100)")

df = None

if data_source == "Use Sample Data":
    df = SAMPLE_DATA.copy()
    st.info("Using built-in sample data (30 students).")

elif data_source == "Upload CSV":
    uploaded = st.file_uploader(
        "Upload CSV with columns: Student, Assignment_1, Assignment_2, Assignment_3, Midterm, Attendance, Final_Grade",
        type=["csv"],
    )
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df)} records from '{uploaded.name}'.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.warning("Please upload a CSV file to continue.")
        with st.expander("📄 Download sample CSV template"):
            csv_bytes = SAMPLE_DATA.to_csv(index=False).encode()
            st.download_button("Download Template", csv_bytes, "grade_template.csv", "text/csv")

else:
    st.subheader("⌨️ Enter Student Records")
    n_students = st.number_input("Number of students", min_value=5, max_value=200, value=10)

    col_headers = ["Student", "Assignment_1", "Assignment_2", "Assignment_3", "Midterm", "Attendance", "Final_Grade"]
    template = pd.DataFrame(
        {
            "Student": [f"S{i+1:03d}" for i in range(int(n_students))],
            "Assignment_1": [75.0] * int(n_students),
            "Assignment_2": [75.0] * int(n_students),
            "Assignment_3": [75.0] * int(n_students),
            "Midterm":      [70.0] * int(n_students),
            "Attendance":   [85.0] * int(n_students),
            "Final_Grade":  [0.0]  * int(n_students),
        }
    )
    edited = st.data_editor(template, num_rows="dynamic", use_container_width=True)
    if st.button("Use This Data"):
        df = edited.copy()
        st.success("Data accepted.")

if df is not None and not df.empty:
    required_cols = set(selected_features) | {"Final_Grade"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Missing columns in data: {missing}")
        st.stop()

    df = df.dropna(subset=list(required_cols))
    if len(df) < 10:
        st.warning("Need at least 10 valid records. Showing data but skipping model training.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "🧠 Model & Predictions", "📊 Visualisations", "🪄 Predict New Student"])

    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Students", len(df))
        c2.metric("Avg Final Grade", f"{df['Final_Grade'].mean():.1f}")
        c3.metric("Pass Rate (≥50)", f"{(df['Final_Grade'] >= 50).mean()*100:.1f}%")
        c4.metric("Avg Attendance", f"{df['Attendance'].mean():.1f}%" if "Attendance" in df.columns else "N/A")

        st.subheader("Descriptive Statistics")
        st.dataframe(df[selected_features + ["Final_Grade"]].describe().round(2), use_container_width=True)

        st.subheader("Correlation Matrix")
        corr_cols = selected_features + ["Final_Grade"]
        corr = df[corr_cols].corr().round(3)
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Feature Correlation Heatmap",
        )
        fig_corr.update_layout(height=420)
        st.plotly_chart(fig_corr, use_container_width=True)

    if len(df) >= 10:
        X = df[selected_features].values
        y = df["Final_Grade"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_sc, y_train)

        y_pred_test  = model.predict(X_test_sc).clip(0, 100)
        y_pred_all   = model.predict(scaler.transform(X)).clip(0, 100)

        mse  = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred_test)

        with tab2:
            st.subheader("Model Performance")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("R² Score", f"{r2:.4f}")
            mc2.metric("RMSE", f"{rmse:.2f}")
            mc3.metric("Training Samples", len(X_train))
            mc4.metric("Test Samples", len(X_test))

            st.subheader("Feature Coefficients (Scaled)")
            coef_df = pd.DataFrame({
                "Feature": selected_features,
                "Coefficient": model.coef_.round(4),
            }).sort_values("Coefficient", key=abs, ascending=False)

            fig_coef = px.bar(
                coef_df, x="Feature", y="Coefficient",
                color="Coefficient",
                color_continuous_scale="RdBu",
                title="Linear Regression Coefficients",
                text="Coefficient",
            )
            fig_coef.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_coef.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig_coef, use_container_width=True)

            st.subheader("Prediction Results (Test Set)")
            results_df = df.iloc[list(range(len(df)))].copy()
            results_df = results_df.reset_index(drop=True)
            results_df["Predicted_Grade"] = np.round(y_pred_all, 1)
            results_df["Error"] = np.round(results_df["Final_Grade"] - results_df["Predicted_Grade"], 2)

            def grade_letter(g):
                if g >= 90: return "A"
                if g >= 80: return "B"
                if g >= 70: return "C"
                if g >= 60: return "D"
                return "F"

            results_df["Actual_Letter"]    = results_df["Final_Grade"].apply(grade_letter)
            results_df["Predicted_Letter"] = results_df["Predicted_Grade"].apply(grade_letter)

            show_cols = ["Student"] + selected_features + ["Final_Grade", "Predicted_Grade", "Error", "Actual_Letter", "Predicted_Letter"] if "Student" in results_df.columns else selected_features + ["Final_Grade", "Predicted_Grade", "Error", "Actual_Letter", "Predicted_Letter"]
            st.dataframe(results_df[show_cols], use_container_width=True)

            csv_out = results_df.to_csv(index=False).encode()
            st.download_button("📥 Download Predictions CSV", csv_out, "predictions.csv", "text/csv")

        with tab3:
            st.subheader("Actual vs Predicted Grades")
            scatter_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": y_pred_test,
            })
            min_val = float(min(scatter_df.min().min() - 5, 0))
            max_val = float(max(scatter_df.max().max() + 5, 100))

            trend_x = np.array([scatter_df["Actual"].min(), scatter_df["Actual"].max()])
            trend_coef = np.polyfit(scatter_df["Actual"], scatter_df["Predicted"], 1)
            trend_y = np.polyval(trend_coef, trend_x)

            fig_avp = px.scatter(
                scatter_df, x="Actual", y="Predicted",
                title="Actual vs Predicted Final Grades (Test Set)",
                labels={"Actual": "Actual Grade", "Predicted": "Predicted Grade"},
                color_discrete_sequence=["#3B82F6"],
            )
            fig_avp.add_trace(go.Scatter(
                x=trend_x, y=trend_y,
                mode="lines", line=dict(color="#F59E0B", width=2), name="Trend"
            ))
            fig_avp.add_shape(
                type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                line=dict(color="red", dash="dash", width=2),
            )
            fig_avp.add_annotation(x=max_val-5, y=max_val-2, text="Perfect Prediction", showarrow=False, font=dict(color="red"))
            st.plotly_chart(fig_avp, use_container_width=True)

            st.subheader("Residual Distribution")
            residuals = y_test - y_pred_test
            fig_res = make_subplots(rows=1, cols=2, subplot_titles=["Residuals vs Predicted", "Residual Histogram"])

            fig_res.add_trace(
                go.Scatter(x=y_pred_test, y=residuals, mode="markers",
                           marker=dict(color="#F59E0B", opacity=0.7), name="Residuals"),
                row=1, col=1
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            fig_res.add_trace(
                go.Histogram(x=residuals, nbinsx=12, marker_color="#8B5CF6", name="Frequency"),
                row=1, col=2
            )
            fig_res.update_layout(height=380, showlegend=False, title_text="Residual Analysis")
            st.plotly_chart(fig_res, use_container_width=True)

            st.subheader("Score Distribution by Component")
            dist_cols = selected_features + ["Final_Grade"]
            fig_dist = go.Figure()
            colors = px.colors.qualitative.Plotly
            for i, col in enumerate(dist_cols):
                fig_dist.add_trace(go.Box(y=df[col], name=col, marker_color=colors[i % len(colors)]))
            fig_dist.update_layout(title="Score Distributions", yaxis_title="Score", height=400)
            st.plotly_chart(fig_dist, use_container_width=True)

            st.subheader("Grade Distribution")
            grade_counts = results_df["Predicted_Letter"].value_counts().reindex(["A","B","C","D","F"], fill_value=0)
            fig_pie = px.pie(
                values=grade_counts.values,
                names=grade_counts.index,
                title="Predicted Grade Distribution",
                color_discrete_sequence=px.colors.sequential.RdBu,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            if len(selected_features) >= 2:
                st.subheader("Feature Scatter Matrix")
                fig_matrix = px.scatter_matrix(
                    df[selected_features + ["Final_Grade"]],
                    dimensions=selected_features[:4],
                    color=df["Final_Grade"],
                    color_continuous_scale="Viridis",
                    title="Pairwise Feature Scatter (coloured by Final Grade)",
                )
                fig_matrix.update_layout(height=550)
                st.plotly_chart(fig_matrix, use_container_width=True)

        with tab4:
            st.subheader("🪄 Predict Grade for a New Student")
            st.markdown("Enter scores below to get an instant grade prediction.")

            new_vals = {}
            pred_cols = st.columns(len(selected_features))
            for i, feat in enumerate(selected_features):
                min_v = float(df[feat].min())
                max_v = float(df[feat].max())
                mean_v = float(df[feat].mean())
                with pred_cols[i]:
                    new_vals[feat] = st.number_input(
                        feat.replace("_", " "), min_value=0.0, max_value=100.0,
                        value=round(mean_v, 1), step=0.5
                    )

            if st.button("Predict Final Grade", type="primary", use_container_width=True):
                input_arr = np.array([[new_vals[f] for f in selected_features]])
                input_sc  = scaler.transform(input_arr)
                predicted  = float(model.predict(input_sc)[0])
                predicted  = max(0.0, min(100.0, predicted))
                letter = grade_letter(predicted)

                color_map = {"A": "green", "B": "blue", "C": "orange", "D": "red", "F": "red"}
                st.markdown(f"""
                <div style="padding:20px; border-radius:12px; background:#1e293b; text-align:center; margin-top:12px;">
                    <h2 style="color:#94a3b8; margin:0;">Predicted Final Grade</h2>
                    <h1 style="font-size:64px; margin:8px 0; color:#f1f5f9;">{predicted:.1f}</h1>
                    <h2 style="font-size:36px; color:{color_map.get(letter,'white')}; margin:0;">Grade: {letter}</h2>
                </div>
                """, unsafe_allow_html=True)

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=predicted,
                    title={"text": "Predicted Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#3B82F6"},
                        "steps": [
                            {"range": [0, 50],  "color": "#FEE2E2"},
                            {"range": [50, 60], "color": "#FEF9C3"},
                            {"range": [60, 70], "color": "#E0F2FE"},
                            {"range": [70, 80], "color": "#DCFCE7"},
                            {"range": [80, 90], "color": "#BBF7D0"},
                            {"range": [90, 100],"color": "#86EFAC"},
                        ],
                        "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50},
                    }
                ))
                fig_gauge.update_layout(height=320)
                st.plotly_chart(fig_gauge, use_container_width=True)

                with st.expander("📋 Input Summary"):
                    inp_df = pd.DataFrame([new_vals])
                    inp_df["Predicted_Grade"] = round(predicted, 1)
                    inp_df["Grade_Letter"] = letter
                    st.dataframe(inp_df, use_container_width=True)

    else:
        with tab2:
            st.warning("Not enough data to train the model. Please provide at least 10 student records.")
        with tab3:
            st.warning("Visualisations will appear once the model is trained.")
        with tab4:
            st.warning("Prediction will be available once the model is trained.")
