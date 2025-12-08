#import "@preview/charged-ieee:0.1.4": ieee


#set page(numbering: "– 1 of 1 –")

#show: ieee.with(
  title: [#text("Smooth and Nonsmooth Markowitz Portfolio Optimization", size: 20pt)],
  authors: (
    (
      name: "Lucas Ahou",
    ),
    (
      name: "Guerand Dewell",
    ),
  ),
  index-terms: ("Scientific writing", "Typesetting", "Document creation", "Syntax"),
  bibliography: bibliography("refs.yml"),
  figure-supplement: [Fig.],
)


//////////////////////
// Page Configuration
/////////////////////
#set text(hyphenate: false)

//////////////////////
// Math Configuration
/////////////////////
#let nonumeq(eq) = math.equation(block: true, numbering: none, eq)

= Introduction

Portfolio optimization is the core of modern quantitative finance. It helps investor solve the trade-off problem between maximizing expected returns from investments while managing risks. Originally, this problem was described as a smooth mean-variance problem introduced by Markowitz @markowitz1952. This problem minimizes the portfolio variance while targeting a certain expected return. This first problem, however, does not take _transaction cost_ into account, therefore leading to a second version of this problem integrating this in the objective. This new problem is more realistic, though it is now non-smooth. Because of this key difference between the two models, the suitable optimization methods will differ from one another. 

The aim of this report is thus to compare different methods for both models by analyzing the computational cost of the methods and their convergence--both empirically and using theoretical results-- 

= Data

As said in the introduction, we will work on historical data sampled from the `S&P500` index. The dataset is described in a `.csv` file where we have access to the name of the different stocks, the date and the open, close and low/high prices at that date. We are actually only interested in the date, name and close prices in this dataset. In fact we are actually interested in finding the average return vector $mu$ and the corresponding covariance matrix of returns $Sigma$. The first step was to extract the close prices as a matrix where each row corresponds to a different date and each column corresponds to a different stock's name. We then had to compute the returns at each date. At a time $t$, the return $r_t$ is given by:


#nonumeq($r_t = (p_t - p_(t-1))\/p_(t-1)$)


The return $mu$ is then simply given by the average of the returns with respect to time, for each assets. The covariance matrix $Sigma$ is also simply given by the covariance matrix of the return matrix. A strong property of (sample) covariance matrices is that they are always square, symmetric and positive semi-definite. This property is mandatory to be able to use strong theoretical results for convergence because it makes our objective function convex (quadratic function with PSD matrix).

In our implementation, we also included a feature to select only $n$ stocks within all the available stocks. This will allow us to compare the computational costs of the diverse methods as the dimension of the variable increases. 

= Smooth Model

// What does the model mean?

// How could you choose the parameter $lambda$? What does it mean to choose a smaller/larger value?

// Is there any additional interesting information on the model?

// Is there any additional interesting information on the
// model? Description of the methods
// What do you need for each of them?
// Compare the theory with some first numerical results.
// What can be improved compared to the theory? Why?
// Is it normal?

The first model we will study is the _Smooth Markowitz model_. In this section, we will first present the formulation of the model and the meaning of its constituting parts. We will then present the three methods we will implement to solve this problem and that we will analyze later.

== Model

The _Smooth Markowitz model_ is mean-variance problem that was introduced by Markowitz@markowitz1952 in 1952. It is defined as follows:

$
  min_(w in Delta) f(w) = 1/2 w^top Sigma w - lambda w^top mu
$

where $w in RR^n$, $Sigma in RR^n$ and $mu in RR^n$.\
The vector of variables $w$ represents the _weights_ of our portfolio, i.e. the proportion of each assets that constitutes our portoflio. Because each component $w_i$ of $w$ represents a percentage of our portfolio, they must sum up to one in a realistic scenario. This is why we define the feasible set as the simplex:

#nonumeq($Delta = {w in RR^n: w_i >= 0, bb(1)^top w = 1}$)

The matrix $Sigma$ is the covariance matrix and $mu$ is the average vector of the returns available in the available dataset. Therefore, the model aims to do the following:
1. The first term $1/2 w^top Sigma w$ represents the variance of the portfolio's return. Intuitively, we want to minimze it as this is a measure of risk. In fact, a highly variating portfolio return means we, a lot of time, will encounter negative results. Thus minimizing this term allows to have a more consistent portfolio.
2. The second term $- lambda w^top mu$ is there to target a maximum average return, while still minimizing the variance with the presence of the first term. Without this second term, the solution for this problem will only try to create a consistent portfolio and will thus not make a lot of profit. The $lambda > 0$ constant is a hyperparameter which controls the risk. The bigger it is, the more the model will try to find a portfolio selection that maximizes the average return compared to minimizing the variance. Therefore, this parameter represents the risk we are willing to take before solving this problem. The greater it is, the higher the risk is.  

Now that we have seen an overview of this model and explained its meaning, we will discuss some properties of this model.

First of all, the most important aspect is that the model is convex. In fact, the objective function is a sum between a positive semi-definite quadratic form (first term) and a linear function in the variables (second term). A sum of two convex functions is convex, so the objective is convex. The feasible set is also convex. This can be easily checked with the following computations:
\
$-->$ Let $w, v in Delta = {w in RR^n: w_i >= 0, sum_(i=1)^(n) w_i = 1}$, $gamma in [0, 1]$ and $u := gamma w + (1 - gamma)v$

1. $u_i = gamma w_i + (1 - gamma)v_i >= 0$ because $w_i, v_i, gamma, (1-gamma) >= 0 quad checkmark$

2. $sum_(i) u_i = sum_(i) gamma w_i + (1 - gamma)v_i = gamma sum_(i) w_i + (1-gamma) sum_(i) v_i = gamma + (1-gamma) = 1 quad checkmark$

Thus $u in Delta$ and $Delta$ is convex $square.filled$\
The reason why it is important

== Projected Gradient Descent

Description of the methods

What do you need for each of them?

Compare the theory with some first numerical results.

What can be improved compared to the theory? Why? Is it normal?

For example:

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Gradient Descent],
)[
  #set align(left)
  #block(
    width: 100%,
    inset: 10pt,
    stroke: 0.5pt + black,
  )[
    *Input:* step size $alpha > 0$ \
    *Output:* approximate solution $x_N$
    
    #v(0.5em)
    
    *for* $k = 0, 1, dots, N - 1$ *do* \
    #h(2em) compute a gradient $g_k$ \
    #h(2em) $x_(k+1) = x_k - alpha g_k$ \
    *end for*
  ]
]

I need a gradient and a step size for Algorithm 1 to work, so the gradient is $dots$ and a classical choice of step size is $dots$, so I need to compute $dots$.

== Projected Gradient Descent with Momentum

What is the momentum doing?

== Projected Randomized Coordinate Descent

Is it smart to make deterministic choices for the coordinates? Is the answer the same in theory and in practice? Discuss it.

= Non-smooth Model

== Model

Key differences with the smooth case. What fundamental changes are you expecting? Do you verify them?

== Projected Subgradient Method

== Proximal Gradient Descent

Did you think of other methods? Why could they help solve the problem? What structure of the problem made you think of this method?

= Numerical results

What comparisons could you make between the different methods? Is it always a fair choice?

Can some methods be greatly improved compared to the theory?

Are some of them disappointing? Do you have an explanation?

In general, does a model (smooth or non-smooth) bring better solutions? What do you mean by better? Are the methods faster? Do the solutions have a particular structure? Is it normal?

= Conclusion

Summary.