import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src to python path to import glass_classifier package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from glass_classifier import config, predict, data_loader, visualization, train

st.set_page_config(page_title="Glass Type Classifier", layout="wide")

st.title("🔬 Glass Type Classifier")
st.markdown("Classify glass based on chemical composition using KNN, SVM, Random Forest, and Gradient Boosting.")


# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Prediction", "Data Analysis", "Model Training"])

if page == "Prediction":
    st.header("🔮 Make a Prediction")
    
    # Check if models exist
    knn, svm, rf, gb, lr, scaler = predict.load_models()
    
    if knn is None:
        st.warning("Models not trained yet. Go to 'Model Training' page first.")
    else:
        models_dict = {'knn': knn, 'svm': svm, 'rf': rf, 'gb': gb, 'lr': lr, 'scaler': scaler}


        
        st.subheader("Enter Chemical Properties:")
        
        input_data = []
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ri = st.number_input("RI (Refractive Index)", value=1.51, format="%.5f")
            input_data.append(ri)
            na = st.number_input("Na (Sodium)", value=13.0)
            input_data.append(na)
            mg = st.number_input("Mg (Magnesium)", value=3.0)
            input_data.append(mg)
            
        with col2:
            al = st.number_input("Al (Aluminum)", value=1.5)
            input_data.append(al)
            si = st.number_input("Si (Silicon)", value=72.0)
            input_data.append(si)
            k = st.number_input("K (Potassium)", value=0.5)
            input_data.append(k)
            
        with col3:
            ca = st.number_input("Ca (Calcium)", value=8.0)
            input_data.append(ca)
            ba = st.number_input("Ba (Barium)", value=0.0)
            input_data.append(ba)
            fe = st.number_input("Fe (Iron)", value=0.0)
            input_data.append(fe)
            
        if st.button("Classify Glass Type"):
            preds = predict.predict_sample(models_dict, input_data)
            
            st.success("Prediction Complete!")
            
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.info(f"**KNN:** Type {preds['KNN']}")
                st.caption(data_loader.get_class_name(preds['KNN']))
            with p2:
                st.info(f"**SVM:** Type {preds['SVM']}")
                st.caption(data_loader.get_class_name(preds['SVM']))
            with p3:
                st.info(f"**RF:** Type {preds['Random Forest']}")
                st.caption(data_loader.get_class_name(preds['Random Forest']))
            with p4:
                st.info(f"**GB:** Type {preds['Gradient Boosting']}")
                st.caption(data_loader.get_class_name(preds['Gradient Boosting']))
            
            st.info(f"**LR:** Type {preds['Logistic Regression']}")
            st.caption(data_loader.get_class_name(preds['Logistic Regression']))



elif page == "Data Analysis":
    st.header("📊 Data Analysis")
    try:
        df = data_loader.load_data()
        st.write("### Dataset Preview")
        st.dataframe(df.head())
        
        st.write("### Statistics")
        st.write(df.describe())
        
        st.write("### Correlations")
        fig_corr = visualization.plot_correlation_matrix(df)
        st.pyplot(fig_corr)
        
        st.write("### Class Distribution")
        fig_dist = visualization.plot_class_distribution(df['Type'])
        st.pyplot(fig_dist)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")

elif page == "Model Training":
    st.header("⚙️ Model Training")
    
    if st.button("Train Models"):
        with st.spinner("Training models..."):
            try:
                artifacts = train.train_pipeline(save_models=True)
                st.success("Training completed successfully!")
                st.write("Models saved to `models/` directory.")
                
                # Show test metrics (evaluate on the fly)
                from glass_classifier import evaluate
                knn_eval = evaluate.evaluate_model(artifacts['knn'], artifacts['X_test_scaled'], artifacts['y_test'], "KNN")
                svm_eval = evaluate.evaluate_model(artifacts['svm'], artifacts['X_test_scaled'], artifacts['y_test'], "SVM")
                rf_eval = evaluate.evaluate_model(artifacts['rf'], artifacts['X_test_scaled'], artifacts['y_test'], "Random Forest")
                gb_eval = evaluate.evaluate_model(artifacts['gb'], artifacts['X_test_scaled'], artifacts['y_test'], "Gradient Boosting")
                lr_eval = evaluate.evaluate_model(artifacts['lr'], artifacts['X_test_scaled'], artifacts['y_test'], "Logistic Regression")
                
                st.subheader("Model Performance")
                c1, c2 = st.columns(2)
                with c1:
                    st.write("### Accuracy Scores")
                    metrics_df = pd.DataFrame({
                        'Model': ['KNN', 'SVM', 'Random Forest', 'Gradient Boosting', 'Logistic Regression'],
                        'Accuracy': [
                            knn_eval['accuracy'], 
                            svm_eval['accuracy'],
                            rf_eval['accuracy'],
                            gb_eval['accuracy'],
                            lr_eval['accuracy']
                        ]
                    })

                    st.dataframe(metrics_df)
                    
                with c2:
                    st.write("### Feature Importance (Random Forest)")
                    fig_imp = visualization.plot_feature_importance(artifacts['rf'], config.FEATURE_COLUMNS)
                    if fig_imp:
                        st.pyplot(fig_imp)

                st.subheader("Confusion Matrices")
                cm1, cm2, cm3, cm4 = st.columns(4)
                with cm1:
                    st.pyplot(visualization.plot_confusion_matrix(knn_eval['confusion_matrix'], "KNN"))
                with cm2:
                    st.pyplot(visualization.plot_confusion_matrix(svm_eval['confusion_matrix'], "SVM"))
                with cm3:
                    st.pyplot(visualization.plot_confusion_matrix(rf_eval['confusion_matrix'], "RF"))
                with cm4:
                    st.pyplot(visualization.plot_confusion_matrix(gb_eval['confusion_matrix'], "GB"))

                    
            except Exception as e:
                st.error(f"Training failed: {e}")
