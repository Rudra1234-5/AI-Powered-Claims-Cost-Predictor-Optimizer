        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": user_question}
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        content = response.choices[0].message.content
        st.markdown("### 🧠 GPT-Generated Forecast Code")
        st.code(content)

        code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            exec_globals = {
                "pd": pd,
                "Prophet": Prophet,
                "plot_plotly": plot_plotly,
                "df": df,
                "st": st,
                "px": px
            }
            exec(code, exec_globals)
        else:
            st.warning("⚠️ No Python code block found in the response.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
