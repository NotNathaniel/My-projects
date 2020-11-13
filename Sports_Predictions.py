import pandas as pd
import numpy as np
from scipy.stats import norm, truncnorm, multivariate_normal
from matplotlib import pyplot as plt
import time


def sample_s(m1, m2, sigma1, sigma2, sigma3, t):
    div=(sigma2**(-1)+sigma3**(-1))*(sigma1**(-1)+sigma3**(-1))-sigma3**(-2)
    sigma_st=np.array([[sigma2**(-1)+sigma3**(-1), sigma3**(-1)],
               [sigma3**(-1), sigma1**(-1)+sigma3**(-1)]])/div
    multiplier=np.array([[sigma1**(-1)*m1+sigma3**(-1)*t],[sigma2**(-1)*m2-sigma3**(-1)*t]])
    mu_st=np.matmul(sigma_st,multiplier)
    # sample s1, s2
    return multivariate_normal.rvs(mean=mu_st.flatten(), cov=sigma_st)


# compare question 2 in the quiz
def sample_t(s1, s2, y, sigma_t):
    if y == 1:
        return truncnorm.rvs(0, np.inf, loc=(s1-s2), scale=np.sqrt(sigma_t))
    else:
        return truncnorm.rvs(np.NINF, 0, loc=(s1-s2), scale=np.sqrt(sigma_t))


# used to predict new games
def marginal_y(mu1, mu2, sigma1, sigma2, sigmat):
    return 1-norm.cdf(0,mu1-mu2, np.sqrt(sigma1+sigma2+sigmat))


# calculate the marginal of y includeing the data bias
def marginal_y_biased(mu1, mu2, sigma1, sigma2, sigmat):
    # 0.3 standard deviations equals to 11.79%
    offset_mu = 0.25 * np.sqrt(sigma1+sigma2+sigmat)
    return 1-norm.cdf(0,mu1-mu2+offset_mu, np.sqrt(sigma1+sigma2+sigmat))


def Gauss_mult(m1, m2, s1, s2):
    s = 1 / (1 / s1 + 1 / s2)
    m = (m1 / s1 + m2 / s2) * s
    return m, s


def Gauss_div(m1, m2, s1, s2):
    m, s = Gauss_mult(m1, m2, s1, -s2)
    return m, s


def Gauss_trunc(a, b, m1, s1):
    b_norm = (b - m1) / np.sqrt(s1)
    a_norm = (a - m1) / np.sqrt(s1)
    m= truncnorm.mean(a_norm, b_norm, loc=m1,scale=np.sqrt(s1))
    s=truncnorm.var(a_norm,b_norm,loc=m1,scale=np.sqrt(s1))
    return m, s


# the gibbs sampler should return mean1, std1, mean2 and std2 for the two players
# burn in and number of samples empirically chosen
def gibbs_sampler(prior_1, prior_2, y, sigma_t=2, t_=50, b=20, K=1000):
    # lists to store the samples
    s1 = []
    s2 = []

    # here is where the magic happens
    for i in range(K):
        s = sample_s(prior_1[0], prior_2[0], prior_1[1], prior_2[1], sigma_t, t_)
        s1_ = s[0]
        s1.append(s1_)
        s2_ = s[1]
        s2.append(s2_)
        t_ = sample_t(s1_, s2_, y, sigma_t)[0]

    # return the sample values without the burn period
    return s1[b:K], s2[b:K]


# Q4.2
def estimate_posterior(prior_1, prior_2, y, sigma_t=2):
    s1, s2 = gibbs_sampler(prior_1, prior_2, y, sigma_t=sigma_t)
    # estimate mean and std for the two players
    s1_m = estimate_mean(s1)
    s1_s = estimate_std(s1, s1_m)
    s2_m = estimate_mean(s2)
    s2_s = estimate_std(s2, s2_m)
    # return results posterior_1 and posterior_2
    return (s1_m, s1_s), (s2_m, s2_s)


# Q4.2
def estimate_mean(X):
    return 1/len(X) * sum(X)


def estimate_std(X, mean):
    return 1/len(X) * sum([(x - mean) ** 2 for x in X])


def q4():
    prior = (0.0, 1.0)
    k = 100
    s1, s2 = gibbs_sampler(prior, prior, y=1, b=0, K=k)
    s1_m = estimate_mean(s1)
    s1_s = estimate_std(s1, s1_m)
    s2_m = estimate_mean(s2)
    s2_s = estimate_std(s2, s2_m)
    plt.plot(range(0, k), s1,
             range(0, k), s2)
    plt.show()
    sigma_t = 2
    k = 100
    s1, s2 = gibbs_sampler((s1_m, s1_s), (s2_m, s2_s), 1, b=0, K=k)
    plt.plot(range(0, k), s1,
             range(0, k), s2)
    plt.show()
    prior = (0.0, 1.0)
    # test out different k values and for each plot the distribution and the estimated gaussian
    for k_ in [50, 100, 150, 500, 1000, 5000, 10000, 25000]:
        start = time.time()
        s1, s2 = gibbs_sampler(prior, prior, y=1, K=(k_+20))
        s1_m = estimate_mean(s1)
        s1_s = estimate_std(s1, s1_m)
        s2_m = estimate_mean(s2)
        s2_s = estimate_std(s2, s2_m)
        end = time.time()
        elapsed = end - start
        x = np.linspace(-5, 5, 100)
        plt.plot(x, norm.pdf(x, s1_m, s1_s), label="posterior s1")
        plt.hist(s1, density=1, label="samples gibbs")
        plt.xlabel("Number of gibbs samples: {}, Required time {:.2f} s".format(k_, elapsed))
        plt.legend()
        plt.show()
    prior = (0.0, 1.0)
    # run gibbs sampler and estimate posterior gaussians
    s1, s2 = gibbs_sampler(prior, prior, y=1, K=1000)
    s1_m = estimate_mean(s1)
    s1_s = estimate_std(s1, s1_m)
    s2_m = estimate_mean(s2)
    s2_s = estimate_std(s2, s2_m)
    # plot 4 gaussian distributions
    x = np.linspace(-5, 5, 100)
    plt.plot(x, norm.pdf(x, 0, 1), label="prior")
    plt.plot(x, norm.pdf(x, s1_m, s1_s), label="posterior s1")
    plt.plot(x, norm.pdf(x, s2_m, s2_s), label="posterior s2")
    plt.legend()
    plt.show()


def output_result(x):
    # output the binary result based on a winning probability
    if  x >= 0.5:
        y_ = 1
    else:
        y_ = -1
    return y_


def get_y(row):
    if row["score1"] == row["score2"]:
        return 0
    elif row["score1"] > row["score2"]:
        return 1
    else:
        return -1


def q56(shuffle=False, gibbs=True, advanced_predictor=False):
    print("Using Seria A data")
    data = pd.read_csv('SerieA.csv')
    # create new row with y values
    data['y'] = data.apply(lambda x: get_y(x), axis=1)
    # filter out the games where the teams draw
    data = data.loc[data['y'] != 0]
    # create a list of all the teams and store their skill representation
    d_teams = data['team1'].unique()
    teams = pd.DataFrame(index=d_teams)
    # set starting values for the mean and the std of the skill representation
    teams['mean'] = 0.0
    teams['std'] = 1.0
    if shuffle:
        print("Shuffle data in advance for Q.5")
        data = data.sample(frac=1).reset_index(drop=True)
    correct_predictions = 0
    random_predictions = 0
    # run through all the matches
    for index, row in data.iterrows():
        # fetch current values for team1 and team2
        s1 = teams.loc[row['team1']]
        s2 = teams.loc[row['team2']]
        # get priors
        prior_1 = (s1['mean'], s1['std'])
        prior_2 = (s2['mean'], s2['std'])
        sigma_t = 2
        # Q6: predict who should win based on the model
        if advanced_predictor:
            y_ = output_result(marginal_y_biased(prior_1[0], prior_2[0], prior_1[1], prior_2[1], sigma_t))
        else:
            y_ = output_result(marginal_y(prior_1[0], prior_2[0], prior_1[1], prior_2[1], sigma_t))
        # fetch the actual result
        y = row['y']
        if y == 1:
            random_predictions += 1
        if y == y_:
            correct_predictions += 1
        # run sampling
        if gibbs:
            posterior_1, posterior_2 = estimate_posterior(prior_1, prior_2, y, sigma_t=sigma_t)
        else:
            posterior_1, posterior_2 = moment_matching(prior_1, prior_2, y, st=sigma_t)
        # rewrite values mean and std
        s1['mean'] = posterior_1[0]
        s1['std'] = posterior_1[1]
        s2['mean'] = posterior_2[0]
        s2['std'] = posterior_2[1]

    if advanced_predictor:
        print("Using advanced predictor")
    if gibbs:
        print("Using gibbs sampling")
    else:
        print("Using moment matching")
    print(f"The prediction rate is {correct_predictions / data.shape[0]}\nUsing random guessing it is {random_predictions / data.shape[0]}")
    teams = teams.sort_values(["mean", "std"], ascending=False)
    x = np.linspace(-10, 10, 100)
    for num, (index, row) in enumerate(teams.iterrows()):
        l = str(num + 1) + ". " + index
        plt.plot(x, norm.pdf(x, row['mean'], row['std']), label=l)
    plt.legend(title="Ranking", loc="upper left", bbox_to_anchor=(1,1), fontsize='x-small')
    plt.show()


def moment_matching(prior1, prior2, y, st=2):
    m1 = prior1[0]
    s1 = prior1[1]
    m2 = prior2[0]
    s2 = prior2[1]
    #we call this c
    mus2_ft_m=m2
    mus2_ft_s=s2
    mus1_ft_m=m1
    mus1_ft_s=s1
    muft_t_m=mus1_ft_m-mus2_ft_m
    muft_t_s=mus1_ft_s+mus2_ft_s+st
    # truncated gaussian approximation via moment-matching
    if y == -1:
        a = np.NINF
        b = 0
    else:
        a = 0
        b = 1000
    # q/mufxt_t approximates mut_fxt
    q_m, q_s = Gauss_trunc(a, b, muft_t_m, muft_t_s)
    p_m, p_s = Gauss_div(q_m, muft_t_m , q_s, muft_t_s)
    #see notes
    muft_s1_m=m2+p_m
    muft_s1_s=p_s+st+s2
    s1_m, s1_s = Gauss_mult(muft_s1_m, m1, muft_s1_s, s1)
    muft_s2_m=m1-p_m
    muft_s2_s=p_s+st+s1
    s2_m, s2_s = Gauss_mult(muft_s2_m, m2, muft_s2_s, s2)
    return (s1_m, s1_s), (s2_m, s2_s)


def q78():
    prior = (0.0, 1.0)
    # run gibbs sampler and estimate posterior gaussians
    s1, s2 = gibbs_sampler(prior, prior, y=1, K=1000)
    s1_m = estimate_mean(s1)
    s1_s = estimate_std(s1, s1_m)
    s2_m = estimate_mean(s2)
    s2_s = estimate_std(s2, s2_m)
    # run matching
    s1_, s2_ = moment_matching(prior, prior, y=1)
    # plot 4 gaussian distributions
    x = np.linspace(-4, 4, 100)
    plt.plot(x, norm.pdf(x, 0, 1), label="prior")
    plt.plot(x, norm.pdf(x, s1_m, s1_s), label="posterior s1 gibbs")
    plt.plot(x, norm.pdf(x, s1_[0], s1_[1]), label="posterior s1 moment-matching")
    # plt.plot(x, norm.pdf(x, s2_m, s2_s), label="posterior s2")
    plt.legend()
    plt.show()
    plt.plot(x, norm.pdf(x, 0, 1), label="prior")
    plt.plot(x, norm.pdf(x, s1_[0], s1_[1]), label="posterior s1 moment-matching")
    plt.plot(x, norm.pdf(x, s2_[0], s2_[1]), label="posterior s2 moment-matching")
    plt.legend()
    plt.show()


def q9():
    print("Using the ATP dataset")
    data = pd.read_csv('ATP.csv')
    # only consider events from the last 5 years
    data = data.loc[data["tourney_date"] > 20150000, ["winner_id", "winner_name", "loser_id", "loser_name", "tourney_date"]]
    d_teams = np.unique(np.append(data['winner_id'].unique(), data['loser_id'].unique()))
    teams = pd.DataFrame(index=d_teams)
    # set starting values for the mean and the std of the skill representation
    teams['mean'] = 0.0
    teams['std'] = 1.0
    correct_predictions = 0
    # run through all the matches
    for index, row in data.iterrows():
        # Switch betwwen Gibbs and moment-matching
        gibbs = False
        # fetch current values for team1 and team2
        s1 = teams.loc[row['winner_id']]
        s2 = teams.loc[row['loser_id']]
        # get priors
        prior_1 = (s1['mean'], s1['std'])
        prior_2 = (s2['mean'], s2['std'])
        sigma_t = 2
        # predict next game
        y_ = output_result(marginal_y(prior_1[0], prior_2[0], prior_1[1], prior_2[1], sigma_t))
        # in this dataet there is no home or away s1 is per definition the winner of the game
        y = 1
        if y == y_:
            correct_predictions += 1
        # run sampling
        if gibbs:
            posterior_1, posterior_2 = estimate_posterior(prior_1, prior_2, y, sigma_t=sigma_t)
        else:
            posterior_1, posterior_2 = moment_matching(prior_1, prior_2, y, st=sigma_t)
        # rewrite values mean and std
        s1['mean'] = posterior_1[0]
        s1['std'] = posterior_1[1]
        s2['mean'] = posterior_2[0]
        s2['std'] = posterior_2[1]
    print(f"The prediction rate is {correct_predictions / data.shape[0]}")


def main():
    q4()
    q56()
    print("\n")
    q56(shuffle=True)
    print("\n")
    q56(gibbs=False)
    print("\n")
    q56(gibbs=False, advanced_predictor=True)
    print("\n")
    q78()
    q9()


if __name__ == '__main__':
    main()
