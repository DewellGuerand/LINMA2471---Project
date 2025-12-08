#import "@preview/charged-ieee:0.1.4": ieee

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
// Text Configuration
/////////////////////
#set text(hyphenate: false)

//////////////////////
// Math Configuration
/////////////////////
#let nonumeq(eq) = math.equation(block: true, numbering: none, eq)

= Introduction

Portfolio optimization is the core of modern quantitative finance. It helps investor solve the trade-off problem between maximizing expected returns from investments while managing risks. Originally, this problem was described as a smooth mean-variance problem introduced by Markowitz @markowitz1952. This problem minimizes the portfolio variance while targeting a certain expected return. This first problem, however, does not take _transaction cost_ into account, therefore leading to a second version of this problem integrating this in the objective. This new problem is more realistic, though it is now non-smooth. Because of this key difference between the two models, the suitable optimization methods will differ from one another. 

The aim of this report is thus to compare different methods for both models by analyzing the cost of the methods, their convergence--both empirically and using theoretical results--and their performance on historical data from the `S&P500`. 

= Data

As said in the introduction, we will work on historical data sampled from the `S&P500` index. The dataset is described in a `.csv` file where we have access to the name of the different stocks, the date and the open, close and low/high prices at that date. We are actually only interested in the date, name and close prices in this dataset. In fact we are actually interested in finding the average return vector $mu$ and the corresponding covariance matrix of returns $Sigma$. The first step was to extract the close prices as a matrix where each row corresponds to a different date and each column corresponds to a different stock's name. We then had to compute the returns at each date. At a time $t$, the return $r_t$ is given by:


#nonumeq($r_t = (p_t - p_(t-1))\/p_(t-1)$)


The return $mu$ is then simply given by the average of the returns with respect to time, for each assets. The covariance matrix $Sigma$ is also simply given by the covariance matrix of the return matrix. A strong property of (sample) covariance matrices is that they are always square, symmetric and positive semi-definite. This property is mandatory to be able to use strong theoretical results for convergence because it makes our objective function convex (quadratic function with PSD matrix).

Another initiative we have taken into account when processing the dataset is the separation between a _train_ and _test_ dataset. This allows us to solve the models on the training dataset and evaluate the performance of the obtained portfolios on a test one. For this, we implemented a feature to split them based on a specified date. All the data before this date will be included in the training dataset and the remaining one will serve to evaluate the portfolio on "future" data. Along with that, we also included a feature to select only $n$ stocks within all the available stocks. This will allow us to compare computational costs of the diverse methods as the dimension of the variable increases. 

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



== Model

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