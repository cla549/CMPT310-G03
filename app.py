import streamlit as st
from ResuMatch_functions import ResuMatch_prediction, sigmoid_percent, pdf_to_string


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

ROLE_DISPLAY_NAMES = {
    "Business Analyst (BA) Resumes": "Business Analyst",
    "Business Intelligence, Business Object Resumes": "Business Intelligence/Object",
    "Datawarehousing, ETL, Informatica Resumes": "Datawarehousing",
    "Java Developers/Architects Resumes": "Java Developer",
    "Network and Systems Administrators Resumes": "Network/Systems Admin",
    "Project Manager Resumes": "Project Manager",
    "Recruiter Resumes": "Recruiter",
    "SQL Developers Resumes": "SQL Developer",
    "Web Developer Resumes": "Web Developer",
}


if analyze_button:
    if resume_file is None:
        st.error("Please upload a Resume.")

    else:
        try:
            cleaned_resume = pdf_to_string(resume_file)
            
            prediction, score_label_list = ResuMatch_prediction(
                "ResuMatch_LSV.pkl",
                cleaned_resume
            )

            all_role_scores =[]

            for score, role in score_label_list:
                percent = sigmoid_percent(score)
                all_role_scores.append(
                    (role, round(percent, 1))
                )

            all_role_scores = sorted(
                all_role_scores,
                key = lambda x : x[1],
                reverse = True
            )

            predicted_role = ROLE_DISPLAY_NAMES.get(
                prediction,
                prediction
            )

            match_score = None

            for role, score in all_role_scores:
                if role == job_role:
                    match_score = score
                    break

            if match_score is None: 
                raise ValueError(
                    f"Selected role '{job_role}' was not found"
                )

            with upload_tab:
                st.success("Resume text extracted successfully.")

            with results_tab:
                st.success("Analysis completed successfully.")

                st.subheader("Compatibility Score")
                st.metric(
                    label="Role Compatibility Score",
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

