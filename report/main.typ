#import "ieee.typ": ieee
#import "components.typ": frame

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
  bibliography: bibliography("refs.yml", full: true),
  figure-supplement: [Fig.],
)


//////////////////////
// Page Configuration
/////////////////////
#set text(hyphenate: false)
#set enum(numbering: "1.")


//////////////////////
// Math Configuration
/////////////////////
#set math.equation(supplement: [Eq.])
#let nonumeq(eq) = math.equation(block: true, numbering: none, eq)


= Introduction

Portfolio optimization is the core of modern quantitative finance. It helps investor solve the trade-off problem between maximizing expected returns from investments while managing risks. Originally, this problem was described as a smooth mean-variance problem introduced by Markowitz @markowitz1952. This problem minimizes the portfolio variance while targeting a certain expected return. This first problem, however, does not take _transaction cost_ into account, therefore leading to a second version of this problem integrating this in the objective. This new problem is more realistic, though it is now non-smooth. Because of this key difference between the two models, the suitable optimization methods will differ from one another. 

The aim of this report is thus to compare different methods for both models by analyzing the computational cost of the methods and their convergence, both empirically and using theoretical results.

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
$ <eq:smooth>


where $w in RR^n$, $Sigma in RR^n$ and $mu in RR^n$.\
The vector of variables $w$ represents the _weights_ of our portfolio, i.e. the proportion of each assets that constitutes our portoflio. Because each component $w_i$ of $w$ represents a percentage of our portfolio, they must sum up to one in a realistic scenario. This is why we define the feasible set as the simplex:

#nonumeq($Delta = {w in RR^n: w_i >= 0, bb(1)^top w = 1}$)

The matrix $Sigma$ is the covariance matrix and $mu$ is the average vector of the returns available in the available dataset. Therefore, the model aims to do the following:
1. The first term $1/2 w^top Sigma w$ represents the variance of the portfolio's return. Intuitively, we want to minimze it as this is a measure of risk. In fact, a highly variating portfolio return means we, a lot of time, will encounter negative results. Thus minimizing this term allows to have a more consistent portfolio.
2. The second term $- lambda w^top mu$ is there to target a maximum average return, while still minimizing the variance with the presence of the first term. Without this second term, the solution for this problem will only try to create a consistent portfolio and will thus not make a lot of profit. The $lambda > 0$ constant is a hyperparameter which controls the risk. The bigger it is, the more the model will try to find a portfolio selection that maximizes the average return compared to minimizing the variance. Therefore, this parameter represents the risk we are willing to take before solving this problem. The greater it is, the higher the risk is.  

Now that we have seen an overview of this model and explained its meaning, we will discuss some properties of this model.

First of all, the most important aspect is that the model is convex. In fact, the objective function is a sum between a positive semi-definite quadratic form (first term) and a linear function in the variables (second term). A sum of two convex functions is convex, so the objective is convex. The feasible set is also convex. This can be easily checked with the following computations:
#figure(
  kind: "proof",
  supplement: [Proof],
  caption: [$Delta$ is convex],
  frame([
    Let $w, v in Delta = {w in RR^n: w_i >= 0, sum_(i=1)^(n) w_i = 1}$, $gamma in [0, 1]$ and $u := gamma w + (1 - gamma)v$ :

    1. $u_i = gamma w_i + (1 - gamma)v_i >= 0$ because $w_i, v_i, gamma, (1-gamma) >= 0 quad checkmark$

    2. $sum_(i) u_i = sum_(i) gamma w_i + (1 - gamma)v_i = gamma sum_(i) w_i + (1-gamma) sum_(i) v_i = gamma + (1-gamma) = 1 quad checkmark$
    
    Thus $u in Delta$ and $Delta$ is convex $square.filled$
    ]
  )
)



The reason why it is important is that the local minima of a convex problem is also *global*. This will ensure that every method that we implement will converge towards a minimum with the same objective value. It will thus not be necessary to compare the performance of the portfolio in terms of its returns. Another important aspect of convex problems is that it allows us to use strong properties of convergence for the different methods. 

Regarding those results and the implementation of the methods, we need to discuss some key properties of this model. More precisely, we need to derive the gradient and the hessian of the objective function, its smoothness constant as well as the projection operator on the simplex

=== Gradient and hessian of $f$
We will simply derive the objective function $f$ with respect to $w$ to obtain the gradient:
#nonumeq(
  $
    gradient f(w) = Sigma w - lambda mu
  $
)
and a second time to get the hessian:
#nonumeq(
  $
    gradient^2 f(w) = Sigma
  $
)

Something to notice is that the computational complexity of a gradient evaluation is $cal(O)(n^2)$ because it is a matrix-vector multiplication. However, in the context of the _coordinate descent_ method, it is not needed to compute the whole gradient as we will see later.

=== Smoothness constant
The smoothness constant of $f$ is needed for the _Projected Gradient Descent_ for example to use a step size that gives us better guarantees of convergence. Here is the derivation of this constant:

#figure(
  kind: "derivation",
  supplement: [Derivation],
  caption: [Smoothness constant],
  frame([
    $forall w, v in RR^n$, we have:

    #nonumeq($
      ||gradient f(w) - gradient f(v)||_2 &= ||Sigma w - Sigma v||_2
      <= ||Sigma|| dot ||w - v||_2\ 
      &= |lambda_(max)(Sigma)| dot ||w - v||_2\
      ==> L = |lambda_(max)(Sigma)| &= lambda_(max)(Sigma)
    $)

    where we used the fact that $Sigma$ is PSD to simplify the last expression.
    ]
  )
)

=== Projection on the simplex
The projection on the simplex is used by every method presented in this report, except for the interior point method. Below, we derive the calculation of the projection on this set:

#figure(
  kind: "derivation",
  supplement: [Derivation],
  caption: [Projection operator on the simplex],
  frame()[
    We want to solve:

    #nonumeq($
      P_(Delta)(v) = arg min_(w in Delta) ||w - v||_(2)^(2)
    $)

    which can be formulated as a constrained quadratic minimization problem:

    #nonumeq($min_(w in RR^n) & 1/2 ||w - v||_(2)^(2) quad "s.t." quad sum_(i=1)^(n) w_i = 1, w_i >= 0$)

    We now define the Lagrangian of this problem as follows: 

    #nonumeq($cal(L)(w, theta, alpha) = &1/2 sum_i (w_i - v_i)^2 - theta(sum_i w_i -1) \ &- sum_i alpha_i w_i$) 

    where $theta in RR$ and $alpha in RR^n$ are the Lagrange multiplier associated with the equality and nonnegativity constraints respectively.

    We now impose the KKT conditions @kkt:
    
    1. Stationarity:
       #nonumeq(
        $(partial cal(L))/(partial w_i) = w_i - v_i - theta - alpha_i = 0 => w_i = v_i + theta + alpha_i$
       )
    2. Primal feasibility:
       #nonumeq(
        $w_i >= 0,quad sum_(i=1)^(n) w_i = 1$
       )

    3. Dual feasibility: 
       #nonumeq($alpha_i >= 0$)

    4. Complementary slackness:
       #nonumeq($w_i alpha_i = 0$)

   and we now have to solve them. First, we notice that from the complementary slackness:

   #nonumeq(
    $
      &"If" w_i > 0, "then" alpha_i = 0 => w_i = v_i + theta\
      &"If" w_i = 0, "then" alpha_i >= 0 => v_i + theta <= 0
    $
   )

   The second case shows that we do not violate any KKT conditions. We thus have:

   #nonumeq(
    $
      w_i = max(v_i + theta, 0)
    $
   )

   To determine $theta$, we use the equality constraint:

   #nonumeq(
    $
      sum_i max(v_i + theta, 0) = 1
    $
   )

   This sum can also be expressed in the following manner:

   #nonumeq(
    $
      sum_(i: v_i + theta > 0) v_i + theta = 1
    $
   )

   If we now define $k$ as the number of indices that satisfy $v_i + theta > 0$ and $v_((i))$ be the sorted components of $v$, we get:

   #nonumeq(
    $
      sum_(i: v_i + theta > 0) v_i + theta = sum_(i = n - k + 1)^(n) v_((i)) + theta = sum_(i = n - k + 1)^(n) v_((i)) + k theta
    $
   )
  ]
)

// Trick to number frame with the same previous number
#counter(figure.where(kind: "derivation")).update(n => n - 1)

#figure(
  kind: "derivation",
  supplement: [Derivation],
  caption: [(cont.)],
  frame()[
    This gives us:
    #nonumeq(
      $
        theta = (1 - sum_(i = n - k + 1)^(n) v_((i)))/(k)
      $
    )
    
    This $k$ also needs to satisfy: 
    #nonumeq(
      $
        v_((n-k)) + theta <= 0 quad "and" quad v_((n-k+1)) + theta > 0
      $
    )

    There is a unique $k$ that satisfies this relation which gives us the correct theta. 
  ]
)

The number $k$--and thus the projection--can be implemented more efficiently using this algorithmic approach:

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Projection computation],
  frame()[
    *Input*: Vector $v$ to project\
    *Output*: $P_(Delta)(v)$\
    *Time complexity:* $cal(O)(n log n)$

    1. Sort $v$ in descending order $=> u_1 >= u_2 >= dots >= u_n$ #v(.5em)
    2. $k <-- max{j in {1, dots, n}: u_j + (1 - sum_(i = 1)^(j) u_j)/(j) > 0}$
    3. $theta <-- 1/k (1 - sum_(i = 1)^(k) u_j)$
    4. *return* $w$ with $w_i = max(v_i - theta, 0)$ 
  ],
  
) <AlgorithmeProjection>

== Projected Gradient Descent

We will now describe the first optimization method we will implemented for this model: the Projected Gradient Descent.\
Given a point $w in Delta$, the idea is simply to perform a gradient descent step: 
#nonumeq($z = w - alpha gradient f(w)$)

where $alpha > 0$ is our step size. However, it is clear that there is no guarantee that this new point $z$ stays in the simplex after the step. We therefore will simply project this point on the simplex using @AlgorithmeProjection:

#nonumeq($w^+ = P_(Delta)(z) = P_(Delta)(w - alpha gradient f(w))$)
The algorithm is described as follows: 

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Projected Gradient Descent],
  frame()[
      *Input:* $w_0, alpha > 0$ \
      *Output:* approximate solution $w_N$
      
      #v(0.5em)
      
      *for* $k = 0, 1, dots, N - 1$ *do* \
      #h(2em) $g_k <-- gradient f(w_k)$ \
      #h(2em) $w_(k+1) <-- P_Delta (w_k - alpha g_k)$ \
      *end for*
  ]
)

Here--as well as for the other methods--multiple stopping criterion can be used. We can either, as described above, stop after a certain number of iterations, or we could stop after reaching a certain precision (either on the objective value as well as the iterates values). Altough we do not have theoretical results that can improve the convergence of the method, we have actually a lower-bound on the number of iterations to reach a certain precision if we take the right step-size. In fact, here we have a smooth objective function $f$ with a smoothness constant $L = lambda_(max)(Sigma)$. In this case , we can take the step size to be:

#nonumeq(
  $
    alpha = 1/L
  $
)

Knowing that our function is also convex, we have the following convergence result:

#nonumeq(
  $
    f(w_k) - f(w_0) <= (L ||w_0 - w^star||^2)/(2k), quad forall k >= 1
  $
)

Hence, our method has a rate of $cal(O)(1\/k)$, which also means that we need $cal(O)(1\/epsilon)$ to reach an $epsilon$-accuracy for the objective value. We indeed confirm our result numerically.

// #figure(
//   image(
//     "../figures/ProjectedGD_true_iteration_complexity.png"
//   )
// )


== Projected Gradient Descent with Momentum

Here we are going to slightly modify the previous algorithm by introducing the notion of momentum.
In our previous algorithm, we were not taking advantage of the gradient of the previous iterates, i.e. $gradient f(w_(k-1)) , gradient f(w_(k-2)),...,gradient f(w_(0))$. We could then introduce a momentum variable
#nonumeq($ m_(k+1) = beta m_k + (1 - beta ) gradient f(w_k), #h(2em) "with" m_0 = 0
$)
The projected momentum iterates becomes for an $w_k in Delta$ 
#nonumeq(
  $
    w_(k+1) = P_Delta (w_k - gamma m_(k+1))
  $
)
with $beta in [0,1]$

Momentum increases the influence of recent gradients while gradually vanishing older ones, which often accelerates convergence.
The more detailed algorithm is described as follows : 

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Projected Gradient Descent with Momentum],
)[
  #set align(left)
  #set par(first-line-indent: 0em)
  #block(
    width: 100%,
    inset: 10pt,
    stroke: 0.5pt + black,
  )[
    *Input:* $w_0, gamma, beta$ \
    *Output:* approximate solution $w_N$
    
    #v(0.5em)
    $m_0 = 0$
    \
    *for* $k = 0, 1, dots, N - 1$ *do* \
    #h(2em) $g_k <-- gradient f(w_k)$ \
    #h(2em) $m_(k+1) = beta m_k + (1 -beta) g_k$ \
    #h(2em) $w_(k+1) = P_Delta (w_k - gamma m_(k+1))$ \
    *end for*
  ]
]

However, with this algorithm (especially with a fixed $beta$), the rate of convergence has the same order $cal(O)(1/k)$ as the projected gradient descent in the worst-case scenario.\
A better version of this algorithm in the context of convex and smooth functions is the _Nesterov's Accelerated Gradient Method_, whose algorithm is described below:

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Nesterov's accelerated Gradient Method],
)[
  #set align(left)
  #set par(first-line-indent: 0em)
  #block(
    width: 100%,
    inset: 10pt,
    stroke: 0.5pt + black,
  )[
    *Input:* $w_0, L$ \
    *Init:* $w_(-1) = w_0, alpha = 1/L, lambda_0 = beta_0 = 0$\
    *Output:* approximate solution $w_N$
    
    *for* $k = 0, 1, dots, N - 1$ *do* \
    #h(2em) $y_k <-- w_k + beta_k (w_k - w_(k-1))$ \
    #h(2em) $w_(k+1) <-- y_k - alpha gradient f(y_k)$\
    #h(2em) $lambda_(k+1) <-- (1 + sqrt(1 + 4 lambda^2_k))/(2), quad beta_(k+1) <-- (lambda_k - 1)/(lambda_(k+1))$ \
    *end for*
  ]
]

This gives us this theoretical bound:

#nonumeq(
  $
    f(y_k) - f^* <= (2L ||w_0 - w^star||^2)/((k+1)^2), quad forall k >= 1
  $
)

So our convergence rate is $cal(O)(1\/k^2)$, which is faster than gradient descent's one. This also implies that the number of iterations is $cal(O)(1\/sqrt(epsilon))$ to reach a precision of $epsilon$ on the objective value.

== Projected Randomized Coordinate Descent

The last optimization model that we have been ask to implement is the Projected Randomized Coordinate Descent.
In this algorithm, rather than computing the full gradient, we are going to compute one component of the gradient. We are then going to perform one gradient descent in this direction. A description of the method can be written as follow. 
Considering $w_k in Delta$ it is then defined by : 
#nonumeq($
           w_(k+1) = P_Delta (w_k - alpha [gradient f(w_k)]_(i_k)) e_(i_k)
         $)
with  $i_k tilde cal(U){1,...,n}$

We directly identify several major issue. The first one occurs when we update our weight $w_(k+1)$. Indeed by proceeding in such way, we only update one component of the vector corresponding to the index $i_k$. Then, projecting back onto the simplex induce a modification of the entire vector.
To fix it we propose 2 options, the first one is the naive one as we are only going to rebalance our weight by modifying an other random weight such that $sum_i w_i = 1$. The second one is a bit more complex, in this version we are also going to take two random weight and trasnfer some value from one to another but this time more cleaverly. We are going to compute their gradient and direct ourself in the direction inducing the biggest variation in term of our objectif function. The update is performed in such a way that the sum of the two selected weights is preserved, ensuring that the iterate remains in the simplex.
The second issue is related to the stopping criterion from our algorithm. From the beginning we only considered the stopping criterion $||w_(k+1) - w_k|| < epsilon$, but in a Randomized Coordinate descent, the modification of the weight is really small so that $||w_(k+1) - w_k||$ is always really small. So for this algorithm we considered a stopping criterion based on the function value,i.e $||f(w_(k+1)) - f(w_k)||< epsilon $. 


Is it smart to make deterministic choices for the coordinates? Is the answer the same in theory and in practice? Discuss it.


== Adaptive step
In this section we present 3 adaptive step that we have implemented in order to improve the performance of our models.

=== Armijo backtracking line search
In this method we are going to search for the step $alpha$ such that : 
#nonumeq(
  $
    f(x- alpha gradient f(x)) <= f(x) - c alpha ||gradient f(x)||^2
  $
)
=== Barzilai-Borwein step size
Here we are going to look for the step $alpha$ which respect this equality : 
#nonumeq($
           alpha_k = (s^T s) / (s^T y) #h(2em) "with" s_(k-1) &= x_k - x_(k-1),\ y_(k-1) &= gradient f(x_k) - gradient f(x_k-1) 
         $)

=== Exact line search
For a quadratic function we know that the optimal step is : 
#nonumeq($
           alpha_k = (||gradient f(w_k)||^2)/(gradient f(w_k)^T Sigma gradient f(w_k))
         $)
== Comparison of the 3 models


= Non-smooth Model

== Model

The second model we will study is a variant of the first one. Its formulation is given by:

$
  min_(w in Delta) f(w) = 1/2 w^top Sigma w - lambda w^top mu + c||w - w_"prev"||_1
$<eq:nonsmooth>

In @eq:nonsmooth, we observe that the objective takes the same form as in @eq:smooth. However, there is an additional non-smooth term. This term extends the previous model by taking transaction costs into account. In realistic scenarios, buying or selling assets implies a cost which is proportional to the amount of traded assets. To represent this in the model, we add a term proportional to the $cal(l)_1$-distance between the current portfolio ($w$) and a reference one ($w_"prev"$):

#nonumeq(
  $
    c||w - w_"prev"||_1
  $
)

This essentially means that we do not want to drastically change the current allocation of our portoflio. In fact, the $cal(l)_1$-norm encourages _sparse rebalancing_. The parameter $c$ controls the trading cost. A small $c$ implies that the portfolio can change drastically compared to a reference one and vice-versa.

The main difference with the smooth model is that we can not compute the gradient explicitly. Instead, we will have to use methods that either use a subgradient, or other optimization techniques.

In the next sections, we will present the three methods we will implement on this model, that is: projected subgradient method, proximal gradient descent and interior point method. For those methods, we will decompose the previous model into a smooth part and a non-smooth part which we will denote by:

#nonumeq(
  $
    &g(w) := 1/2 w^top Sigma w - lambda w^top mu, quad  h(w) := c||w - w_"prev"||_1\
    ==> &f(w) = g(w) + h(w)
  $
)

== Projected Subgradient Method

In the projected subgradient method, instead of using the gradient of the functionwe will try to use what socalled an subgradient that we define as : 
Given a $overline(x) in RR^n$, we say that a vector $f(overline(x))$ is a subgradient of $f$ at $overline(x)$ when : 
#nonumeq($
           f(y) >= f(overline(x)) + <g(overline(x)) , y - overline(x)> #h(2em) forall y in RR^n
         $)

The projected _subdradient_ descent is then defined as (given $w_0$) : 
#nonumeq($
           w_(k+1) = P_Omega (w_k - alpha_k g_k)
         $)

Intuitively for non-smooth function, the gradient may not be well defined everywhere so we generalize the notion of gradient by introducing subgradient so that it becomes exactly defined in every points.     

From what we have seen in the course, for a constant step size, we know that 

== Proximal Gradient Descent

Instead of having to rely on the subgradient of $h$ in our algorithm, we can use the _proximal gradient method_. We define the proximal operator for $h$ given a step-size $t$ by:

#nonumeq(
  $
    "prox"_(t, h)(x) = arg min_(z in RR^n) 1/(2t) ||x - z||_2^2 + h(z)
  $
)

The proximal gradient method is then given by (given $w_0$):

#nonumeq(
  $
    w_k = "prox"_(t_k, h)(w_(k-1) - t_k gradient g(x^(k-1))), quad k = 1, 2, dots
  $
)

This method intuitively means:

- First term in the $"prox"$ operator: find a point $z$ that stays close to the gradient update of $g$
- Second term in the $"prox"$ operator: Also, try to make $h$ small

Even though this looks like we just reformulated the optimization problem, in some special cases the proximal operator has a closed-form expression. In our case, we have to obtain an expression for the proximal operator of the $cal(l)_1$-norm. It can be proven that, for a function $phi(x) = c||x||_1$, the proximal operator is given by:

#nonumeq(
  $
    ["prox"_(t, phi)(x)]_i = cases(x_i - c t quad &"if" x_i > c t,
                                   0 quad &"if" -c t <= x_i <= c t,
                                   x_i + c t quad &"if" x_i < -c t) 
  $
)

In our case, we have $h(w) = phi(w - w_"prev")$. It is easy to adapt the previous formula to this case\
In the unconstrained case, given a fixed step-size $t <= 1/L$ (where $L$ is the smoothness constant of $g$), we have the following convergence result:

#nonumeq(
  $
    f(w_k) - f^star <= (||w_0 - w^star||_2^2)/(2k t)
  $
)

Therefore, this method has a convergence rate of $cal(O)(1\/k)$, or $cal(O)(1\/epsilon)$. However, we have to consider the cost of computing the proximal in practice. Fortunately, we have a variant of this method based on the Nesterov's accelerated gradient method, the _Accelerated Proximal Gradient Method_. This method is described by:

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Accelerated Proximal Gradient Method],
)[
  #set align(left)
  #set par(first-line-indent: 0em)
  #block(
    width: 100%,
    inset: 10pt,
    stroke: 0.5pt + black,
  )[
    *Input:* $w_0, L$ \
    *Init:* $y_(1) = w_0, t_1 = 1$\
    *Output:* approximate solution $w_N$
    
    *for* $k = 0, 1, dots, N - 1$ *do* \
    #h(2em) $x_k <-- "prox"_(1/L, h)(y_k - 1/L gradient f(y_k))$#v(.5em) 
    #h(2em) $t_(k+1) <-- (1 + sqrt(1 + 4t_k^2))/2$\
    #h(2em) $y_(k+1) <-- w_k + ((t_k - 1)/(t_(k+1)))(w_k - w_(k-1))$ \
    *end for*
  ]
]

This time, in the unconstrained case, we have the following convergence result:

#nonumeq(
  $
    f(w_k) - f^star <= (2||w_0 - w^star||_2^2)/(t(k+1)^2)
  $
)

which thus gives us a convergence rate of $cal(O)(1\/k^2)$ or $cal(O)(1\/sqrt(epsilon))$.

== Long-Step Path-Following Interior-Point method
Did you think of other methods? Why could they help solve the problem? What structure of the problem made you think of this method?

= Numerical results

What comparisons could you make between the different methods? Is it always a fair choice?

Can some methods be greatly improved compared to the theory?

Are some of them disappointing? Do you have an explanation?

In general, does a model (smooth or non-smooth) bring better solutions? What do you mean by better? Are the methods faster? Do the solutions have a particular structure? Is it normal?

= Conclusion

Summary.