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


= Introduction

What is the aim of this work?

In this project our goal is to decide how to allocates wealth among different assets in order to balance expedcted return and risk.


What are the main results?

This is a concise introduction to your work. See Section 3 for details.

= Data

*How did you handle the raw data?*

To manage the data, we imported the CSV file via Panda in order to convert it into a data frame. We then transformed the dataset so that each column corresponds to the closing price of stock X and the corresponding row corresponds to the date.

*How did you estimate $mu$ and $Sigma$?*

To estimate stock returns and the covariance matrix, we used the following formula : $r_t = log (C_t/C_(t-1))$ were $r_t$ correspond to the return at time $t$ and $C_t$ correspond to the close time at $t$.

*A positive semi-definite covariance matrix would greatly improve the model. Why? Is it PSD? If not, can you make it naturally?* 

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