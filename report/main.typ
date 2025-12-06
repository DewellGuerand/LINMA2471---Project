#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [A Typesetting System to Untangle the Scientific Writing Process],
  abstract: [
    The process of scientific writing is often tangled up with the intricacies of typesetting, leading to frustration and wasted time for researchers. In this paper, we introduce Typst, a new typesetting system designed specifically for scientific writing. Typst untangles the typesetting process, allowing researchers to compose papers faster. In a series of experiments we demonstrate that Typst offers several advantages, including faster document creation, simplified syntax, and increased ease-of-use.
  ],
  authors: (
    (
      name: "Martin Haug",
      department: [Co-Founder],
      organization: [Typst GmbH],
      location: [Berlin, Germany],
      email: "haug@typst.app"
    ),
    (
      name: "Laurenz Mädje",
      department: [Co-Founder],
      organization: [Typst GmbH],
      location: [Berlin, Germany],
      email: "maedje@typst.app"
    ),
  ),
  index-terms: ("Scientific writing", "Typesetting", "Document creation", "Syntax"),
  bibliography: bibliography("refs.yml"),
  figure-supplement: [Fig.],
)



#v(1em)

= Introduction

What is the aim of this work?

What are the main results?

This is a concise introduction to your work. See Section 3 for details.

= Data

How did you handle the raw data?

How did you estimate $mu$ and $Sigma$?

A positive semi-definite covariance matrix would greatly improve the model. Why? Is it PSD? If not, can you make it naturally?

= Smooth Model

What does the model mean?

How could you choose the parameter $lambda$? What does it mean to choose a smaller/larger value?

Is there any additional interesting information on the model?

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