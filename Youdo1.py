import numpy as np
import plotly.express as px
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_california_housing

def main(verbose: bool = False):
    cal_housing = fetch_california_housing()
    X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    y = cal_housing.target

    if verbose:
        st.dataframe(X)

    df = pd.DataFrame(
        dict(MedInc=X['MedInc'], Price=cal_housing.target))

    st.dataframe(df)

    st.subheader("House Age independent General Model")
    fig = px.scatter(df, x="MedInc", y="Price")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("My Loss Function")
    st.latex(r"L(\beta_0, \beta_1) = \sum_{i=1}^N min((y_i - (\beta_0 + \beta_1 x))^2, \theta^2)")

    x = df['MedInc']
    # Theta is arbitrarily chosen as 3, we can try another values
    theta = pd.DataFrame({'Theta': np.zeros(x.shape[0]) + 3})

    # Tried to take row-wise minimum value of two columns in a dataframe
    if verbose:
        y_hat = pd.DataFrame({"y_hat": y - 0.3 - 0.5 * x})
        st.write(max(y_hat))
        conct = pd.concat([y_hat**2, theta**2], axis=1)
        st.dataframe(conct)
        conct['min'] = conct[['y_hat', 'Theta']].min(axis=1)
        st.dataframe(conct)

    loss, b0, b1 = [], [], []

    # Calculating the loss functions
    for _b0 in np.linspace(-1, 10, 100):
            for i,_b1 in enumerate(np.linspace(-1, 10, 100)):
                if i == 1:
                    b0.append(_b0)
                    b1.append(_b1)
                    y_hat = pd.DataFrame({"y_hat" :y - _b0 - _b1 * x})
                    conct = pd.concat([y_hat**2, theta**2], axis = 1)
                    conct['min'] = conct[['y_hat', 'Theta']].min(axis=1)
                    loss.append(conct["min"].sum())

    # Plotting loss function for changing b0 or b1 values
    l = pd.DataFrame(dict(b0 = b0, b1 = b1, loss = loss))

    fig = px.scatter(l, x = "b0", y = "loss")
    st.plotly_chart(fig, use_container_width=True)

    # Gradient Descent
    beta = np.random.random(2)

    alpha = 10**-7
    print("Starting sgd")
    for j in range(800):
        y_pred = pd.DataFrame({"y_pred": beta[0] + beta[1] * x})
        # I concatanated y and y_pred, because I received some errors otherwise...
        y_pred = pd.concat([y_pred, pd.DataFrame({"y": y})], axis=1)
        y_pred['y_pred_less'] = np.where((y_pred['y_pred'] <= theta['Theta']), True, False)
        y_pred = y_pred[y_pred.y_pred_less == True]

        g_b0 = (-2 * (y_pred['y'] - y_pred['y_pred'])).sum()
        g_b1 = (-2 * (x * (y_pred['y'] - y_pred['y_pred']))).sum()

        print(f"({j}) beta: {beta}, gradient: {round(g_b0, 3)}, {round(g_b1, 3)}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        if np.linalg.norm(beta - beta_prev) < 0.000001:
            print(f"I do early stoping at iteration {j}")
            break




    # Gradient Descent --> Loss function with L2 Norm
    st.subheader("My Loss Function with L2 norm")
    st.latex(r"L(\beta_0, \beta_1) = \sum_{i=1}^N min((y_i - (\beta_0 + \beta_1 x))^2, \theta^2) + \lambda (\beta_0^2 + \beta_1^2)")

    beta = np.random.random(2)
    print("Starting sgd")
    lam = st.slider("Coefficient Multiplier (Lambda)", 0.001, 10., value = 0.1 )
    for j in range(800):
        y_pred = pd.DataFrame({"y_pred": beta[0] + beta[1] * x})
        # I concatanated y and y_pred, because I received some errors otherwise...
        y_pred = pd.concat([y_pred, pd.DataFrame({"y": y})], axis=1)
        y_pred['y_pred_less'] = np.where((y_pred['y_pred'] <= theta['Theta']), True, False)
        y_pred = y_pred[y_pred.y_pred_less == True]

        g_b0 = (-2 * (y_pred['y'] - y_pred['y_pred'])).sum() + 2 * lam * beta[0]
        g_b1 = (-2 * (x * (y_pred['y'] - y_pred['y_pred']))).sum() + 2 * lam * beta[1]

        print(f"({j}) beta: {beta}, gradient: {round(g_b0, 3)}, {round(g_b1, 3)}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        if np.linalg.norm(beta - beta_prev) < 0.000001:
            print(f"I do early stoping at iteration {j}")
            break


if __name__ == '__main__':
    main(st.checkbox("verbosity"))

