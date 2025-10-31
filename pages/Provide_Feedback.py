import streamlit as st 

st.title("User Feedback & Support")

st.markdown("""
### Help Us Improve the Platform

Your feedback is essential for enhancing our spectroscopic analysis tools. Share your experience, report issues, or suggest new features to help us deliver better solutions for the research community.
""")

with st.form(key="feedback_form"):
    name = st.text_input("Name (optional)")
    email = st.text_input("Email (optional)")
    
    feedback_type = st.selectbox(
        "Type of Feedback",
        options=["General Feedback", "Bug Report", "Feature Request", "Question"]
    )
    
    feedback_text = st.text_area(
        "Your Feedback",
        height=150,
        placeholder="Please provide your feedback here..."
    )
    
    rating = st.slider(
        "How would you rate your experience with this application?",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Poor, 5 = Excellent"
    )
    
    submit_button = st.form_submit_button(label="Submit Feedback")
    
    if submit_button:
        if feedback_text:
            st.success("Thank you for your feedback! We appreciate your input.")
            # Here you would typically save the feedback to a database or send an email
            # For now, we'll just display a success message
        else:
            st.error("Please provide feedback before submitting.")

st.divider()
st.write("If you prefer to contact us directly, please email: lesterjess.heyrana@g.msuiit.edu.ph")
