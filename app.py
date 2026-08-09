import streamlit as st
from resume_reader_new import extract_resume_text
from data_cleaning import clean_text
from Resume_Evaluator import evaluate_role, predict_best_role, get_all_role_scores


st.set_page_config(
    page_title = "🤖 ResuMatch",
    page_icon="📄",
    layout = "wide",
)

st.title("ResuMatch")
st.write("Upload a resume and select a job role to analyze compatibility.")

upload_tab, results_tab = st.tabs(["Upload", "Results"])

with upload_tab: 
    st.subheader("Resume and Job Information")

    left_column, right_column = st.columns(2) 

    with left_column:
        resume_file = st.file_uploader(
            "Upload your resume",
            type = ["pdf", "docx", "txt"],
        )

    with right_column:
        job_role = st.selectbox(
            "Select a job role",
            [
                "Business Analyst",
                "Business Intelligence/Object",
                "Datawarehousing",
                "Java Developer",
                "Network/Systems Admin",
                "Project Manager",
                "Recruiter",
                "SQL Developer",
                "Web Developer",
            ],
        )

    analyze_button = st.button(
        "Analyze Compatibility",
        type = "primary",
    )



if analyze_button:
    if resume_file is None:
        st.error("Please upload a Resume.")

    else:
        try:
            resume_text = extract_resume_text(resume_file)
            cleaned_resume = clean_text(resume_text)
            match_score = evaluate_role(
                cleaned_resume,
                job_role
            )
            match_score = round(match_score, 1)

            predicted_role = predict_best_role(cleaned_resume)

            all_role_scores = get_all_role_scores(cleaned_resume)

            with upload_tab:
                st.success("Resume text extracted successfully.")

                with st.expander("Original Resume"):
                    st.text(resume_text[:2000])

                with st.expander("Cleaned Resume"):
                    st.text(cleaned_resume[:2000])


            with results_tab:
                st.success("Analysis completed successfully.")

                st.subheader("Compatibility Score")
                st.metric(
                    label="Role Compatbility Score",
                    value=f"{match_score}%",
                )

                with st.expander("Compatibility Across Job Roles"):
                    for role, score in all_role_scores:
                        st.write(f"{role}: {score}%")

                st.progress(match_score / 100)

                if match_score >= 60:
                    st.success("Overall result: Good Match")
                elif match_score >= 40:
                    st.warning("Overall result: Moderate Match")
                else:
                    st.error("Overall result: Poor Match")

                st.divider()

                role_column, score_column = st.columns(2)

                with role_column:
                    st.subheader("Predicted Job Role")
                    st.info(predicted_role)

                with score_column:
                    st.subheader("Match Summary")
                    st.write(
                        f"""
                    The uploaded resume has a compatibility score of 
                    **{match_score}%** with the selected **{job_role}** role. 

                    The predicted job role is **{predicted_role}**.
                    """
                )

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error(f"Unable to read the resume: {error}")

