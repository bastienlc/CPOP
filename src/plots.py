from src.utils import *
from src.coefficients import *
from matplotlib import pyplot as plt

def get_coeffs_and_phi(y, changepoints_cpop, beta, sigma, h_type = log_segment_cost, scale = 1):
    coeffs_list = []
    phi_coeffs_list = []

    h = h_type(scale = scale)
    coeffs, phi_coeffs = get_segment_coefficients(y = y, t = changepoints_cpop[1]+1, sigma = sigma, h = h, return_phi_prime = True)
    coeffs_list.append(coeffs)
    phi_coeffs_list.append(phi_coeffs)
    y_cumsum, y_linear_cumsum, y_squarred_cumsum = precompute_sums(y)
    for i, t in enumerate(changepoints_cpop[2:]):
        s = changepoints_cpop[(i+2)-1]
        coeffs, phi_coeffs = get_recursive_coefficients(s+1, coeffs_list[-1], y_cumsum, y_linear_cumsum, y_squarred_cumsum, t+1, sigma, beta, h, return_phi_prime = True)
        coeffs_list.append(coeffs)
        phi_coeffs_list.append(phi_coeffs)

    phi_coeffs_list = phi_coeffs_list[::-1]

    phi_list = []
    last_coeffs = coeffs_list[-1]
    a_, b_, c_ = last_coeffs[0], last_coeffs[1], last_coeffs[2]
    phi = -b_/(2*c_)
    phi_list.append(phi)
    for i in range(len(phi_coeffs_list)):
        alpha, gamma = phi_coeffs_list[i][0], phi_coeffs_list[i][1]
        phi = alpha + gamma*phi
        phi_list.append(phi)

    phi_list = phi_list[::-1]

    return coeffs_list, phi_list

def get_approx_from_phi_list(changepoints_cpop, phi_list):
    approx_by_segment = []

    cp1, cp2 = changepoints_cpop[0], changepoints_cpop[1]
    phi1, phi2 = phi_list[0], phi_list[1]

    slope = (phi2 - phi1)/(cp2 + 1)
    approx0 = (np.arange(1, cp2+2))*slope + phi1
    approx_by_segment.append(approx0)

    for i in range(2, len(changepoints_cpop)):
        cp1, cp2 = changepoints_cpop[i - 1], changepoints_cpop[i]
        phi1, phi2 = phi_list[i-1], phi_list[i]
        slope = (phi2 - phi1)/(cp2 - cp1)
        x_vals = np.arange(cp1 + 1, cp2+1)
        approx = slope*(x_vals - cp1) + phi_list[i-1]
        approx_by_segment.append(approx)

    approx = np.concatenate(approx_by_segment)
    return approx

def get_approx(y, changepoints_cpop, beta, sigma, h_type = log_segment_cost, scale = 1, plot = True, stock_name = "Unspecified", X = None):
    coeffs_list, phi_list = get_coeffs_and_phi(y, changepoints_cpop, beta, sigma, h_type = h_type, scale = scale)
    approx = get_approx_from_phi_list(changepoints_cpop, phi_list)
    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('lightgrey')
        ax.patch.set_facecolor('white')
        ax.plot(y, alpha=0.7, label='Original signal')
        ax.plot(approx, color='red', label='Approximation', alpha=0.7)
        ax.scatter(changepoints_cpop, phi_list, color='red', label='Changepoints')
        ax.set_title("Stock : " + stock_name + "\n" + r"$\beta$ = " + str(beta) + r", $\gamma$ = " + str(scale), fontsize=20)
        # set xtick labels
        if X is not None:
            xtick_labels = X
            xtick_labels = [xtick_labels[i] for i in changepoints_cpop]
            ax.set_xticklabels(xtick_labels, fontsize=12)
        ax.set_xlabel('Time', fontsize=15)
        ax.set_ylabel('Price', fontsize=15)
        ax.legend(fontsize=10)
        last_coeffs = coeffs_list[-1]
        cost = compute_costs(last_coeffs.reshape(1, -1))[0]
        text = "Number of changepoints: " + str(len(changepoints_cpop)) + "\n" + "Cost: " + str(round(cost ,2))
        fig.text(0.08, -0.05, text, horizontalalignment='left', wrap=True , fontsize=12)
        ax.grid()
        plt.show()
    # Cost check
    diff = np.diff(changepoints_cpop)
    diff[0] += 1
    manual_cost = np.sum((y - approx)**2) / sigma**2 + (len(changepoints_cpop) - 2)*beta + np.sum(np.log(diff))
    returned_cost = compute_costs(last_coeffs.reshape(1, -1))[0]
    flag = np.isclose(manual_cost, returned_cost)
    if not flag:
        print("Warning: the cost computed manually and the cost returned by the function are not close")
        print("Manual cost: ", manual_cost)
        print("Returned cost: ", returned_cost)
    if plot:
        return approx, fig
    return approx