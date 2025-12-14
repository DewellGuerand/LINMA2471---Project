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
#set math.mat(delim: "[")
#set math.vec(delim: "[")

= Introduction

Portfolio optimization is the core of modern quantitative finance. It helps investor solve the trade-off problem between maximizing expected returns from investments while managing risks. Originally, this problem was described as a smooth mean-variance problem introduced by Markowitz @markowitz1952. This problem minimizes the portfolio variance while targeting a certain expected return. This first problem, however, does not take _transaction cost_ into account, therefore leading to a second version of this problem integrating this in the objective. This new problem is more realistic, though it is now non-smooth. Because of this key difference between the two models, the suitable optimization methods will differ from one another.

The aim of this report is thus to compare different methods for both models by analyzing the computational cost of the methods and their convergence, both empirically and using theoretical results.

= Data

For both models, we will build them based on historical data sampled from the `S&P500` index. The dataset is described in a `.csv` file where we have access to the name of the different stocks, the date and the open, close and low/high prices at that date #footnote([Note that the data only contains business days so two consecutive dates are not necessarily consecutive days.]). We are actually only interested in the date, name and close prices in this dataset. In fact we are actually interested in finding the average return vector $mu$ and the corresponding covariance matrix of returns $Sigma$. The first step was to extract the close prices as a matrix where each row corresponds to a different date and each column corresponds to a different stock's name. We then had to compute the returns at each date. At a time $t$, the return $r_t$ is given by:


#nonumeq($r_t = (p_t - p_(t-1))\/p_(t-1)$)


The return $mu$ is then simply given by the average of the returns with respect to time, for each assets. The covariance matrix $Sigma$ is also simply given by the covariance matrix of the return matrix. A strong property of (sample) covariance matrices is that they are always square, symmetric and positive semi-definite. This property is mandatory to be able to use strong theoretical results for convergence because it makes our objective function convex.

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

The _Smooth Markowitz model_ is a mean-variance problem that was introduced by Markowitz  in 1952@markowitz1952. It is defined as follows:

$
  min_(w in Delta) f(w) = 1/2 w^top Sigma w - lambda w^top mu
$ <eq:smooth>


where $w in RR^n$, $Sigma in RR^(n times n)$ and $mu in RR^n$.\
The vector of variables $w$ represents the _weights_ of our portfolio, i.e. the proportion of each assets that constitutes it. Because each component $w_i$ of $w$ represents a percentage of our portfolio, they must sum up to one in a realistic scenario. This is why we define the feasible set as the simplex:

#nonumeq($Delta = {w in RR^n: w_i >= 0, bb(1)^top w = 1}$)

The matrix $Sigma$ is the covariance matrix and $mu$ is the average vector of the returns available in the available dataset. Therefore, the model aims to do the following:
1. The first term $1/2 w^top Sigma w$ represents the variance of the portfolio's return. Intuitively, we want to minimize it as this is a measure of risk. In fact, a highly variating portfolio return we will often encounter very high returns, as well as very low ones which indicates an unstable portfolio. Thus minimizing this term allows to have more consistent returns.
2. The second term $- lambda w^top mu$ is there to target a maximum average return while still minimizing the variance with the presence of the first term. Without this, the solution for this problem will only try to aim for consistent returns and will thus not make a lot of profit. The $lambda > 0$ constant is a hyperparameter which controls the risk. The bigger it is, the more the model will try to find a portfolio selection that maximizes the average return compared to minimizing the variance. Therefore, this parameter represents the risk we are willing to take before solving this problem. The greater it is, the higher the risk is.

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
  ]),
)



The reason why it is important is that the local minima of a convex problem is also *global*. This will ensure that every method that we implement will converge towards a minimum with the same objective value. It will thus not be necessary to compare the performance of the portfolio in terms of its returns. Another important aspect of convex problems is that it allows us to use strong properties of convergence for the different methods.

Regarding those results and the implementation of the methods, we need to discuss some key properties of this model. More precisely, we need to derive the gradient and the hessian of the objective function, its smoothness constant as well as the projection operator on the simplex

=== Gradient and hessian of $f$
We will simply differentiate the objective function $f$ with respect to $w$ to obtain the gradient:
#nonumeq(
  $
    gradient f(w) = Sigma w - lambda mu
  $,
)
and a second time to get the hessian:
#nonumeq(
  $
    gradient^2 f(w) = Sigma
  $,
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

    #nonumeq(
      $
        ||gradient f(w) - gradient f(v)||_2 & = ||Sigma w - Sigma v||_2
                                              <= ||Sigma|| dot ||w - v||_2 \
                                            & = |lambda_(max)(Sigma)| dot ||w - v||_2 \
              ==> L = |lambda_(max)(Sigma)| & = lambda_(max)(Sigma)
      $,
    )

    where we used the fact that $Sigma$ is PSD to simplify the last expression.
  ]),
)

=== Projection on the simplex
The projection on the simplex is used by every method presented in this report, except for the interior point method. Below, we derive the calculation of the projection on this set:

#figure(
  kind: "derivation",
  supplement: [Derivation],
  caption: [Projection operator on the simplex],
  frame()[
    We want to solve:

    #nonumeq(
      $
        P_(Delta)(v) = arg min_(w in Delta) ||w - v||_(2)^(2)
      $,
    )

    which can be formulated as a constrained quadratic minimization problem:

    #nonumeq($min_(w in RR^n) & 1/2 ||w - v||_(2)^(2) quad "s.t." quad sum_(i=1)^(n) w_i = 1, w_i >= 0$)

    We now define the Lagrangian of this problem as follows:

    #nonumeq($cal(L)(w, theta, alpha) = 1/2 sum_i (w_i - v_i)^2 - theta(sum_i w_i -1) - sum_i alpha_i w_i$)

    where $theta in RR$ and $alpha in RR^n$ are the Lagrange multipliers associated with the equality and nonnegativity constraints respectively.
  ]
)

// Trick to number frame with the same previous number
#counter(figure.where(kind: "derivation")).update(n => n - 1)

#figure(
  kind: "derivation",
  supplement: [Derivation],
  caption: [Projection operator on the simplex (cont.)],
  frame()[
    We now impose the KKT conditions @kkt:

    1. Stationarity:
      #nonumeq(
        $(partial cal(L))/(partial w_i) = w_i - v_i - theta - alpha_i = 0 => w_i = v_i + theta + alpha_i$,
      )
    2. Primal feasibility:
      #nonumeq(
        $w_i >= 0,quad sum_(i=1)^(n) w_i = 1$,
      )

    3. Dual feasibility:
      #nonumeq($alpha_i >= 0$)

    4. Complementary slackness:
      #nonumeq($w_i alpha_i = 0$)

    and we now have to solve them. First, we notice that from the complementary slackness:

    #nonumeq(
      $
        & "If" w_i > 0, "then" alpha_i = 0 => w_i = v_i + theta \
        & "If" w_i = 0, "then" alpha_i >= 0 => v_i + theta <= 0
      $,
    )

    The second case simply shows that we do not violate any KKT conditions. We thus have:

    #nonumeq(
      $
        w_i = max(v_i + theta, 0)
      $,
    )

    To determine $theta$, we use the equality constraint:

    #nonumeq(
      $
        sum_i max(v_i + theta, 0) = 1
      $,
    )

    This sum can also be expressed in the following manner:

    #nonumeq(
      $
        sum_(i: v_i + theta > 0) v_i + theta = 1
      $,
    )

    If we now define $k$ as the number of indices that satisfy $v_i + theta > 0$ and $v_((i))$ be the sorted components of $v$, we get:

    #nonumeq(
      $
        sum_(i: v_i + theta > 0) v_i + theta = sum_(i = n - k + 1)^(n) v_((i)) + theta = sum_(i = n - k + 1)^(n) v_((i)) + k theta
      $,
    )
 
    This gives us:
    #nonumeq(
      $
        theta = (1 - sum_(i = n - k + 1)^(n) v_((i)))/(k)
      $,
    )

    This $k$ also needs to satisfy:
    #nonumeq(
      $
        v_((n-k)) + theta <= 0 quad "and" quad v_((n-k+1)) + theta > 0
      $,
    )

    There is a unique $k$ that satisfies this relation which gives us the correct theta.
  ],
)

The number $k$--and thus the projection--can be implemented more efficiently using this algorithmic approach:

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Projection on the simplex],
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
The complete algorithm is described as follows:

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
  ],
)

Here (as well as for the other methods) multiple stopping criterions can be used. We can either, as described above, stop after a certain number of iterations, or we could stop after reaching a certain precision (either on the objective value as well as the iterate values). Altough we do not have theoretical results that can improve the convergence of the method, we have actually a lower-bound on the number of iterations to reach a certain precision if we take the right step size. In fact, here we have a smooth objective function $f$ with a smoothness constant $L = lambda_(max)(Sigma)$. In this case, we can take the step size to be:

#nonumeq(
  $
    alpha = 1/L
  $,
)

Knowing that our function is also convex, we have the following convergence result:

#nonumeq(
  $
    f(w_k) - f^star <= (L ||w_0 - w^star||^2)/(2k) <= L/k, quad forall k >= 1
  $,
)

where we used the fact that $||w_0 - w^star||^2 <= D^2 = 2$ where $D$ is the diameter of the simplex.\
Hence, our method has a rate of $cal(O)(1\/k)$, which also means that we need $cal(O)(1\/epsilon)$ to reach an $epsilon$-accuracy for the objective value.
We will later confirm this rate numerically.

== Adaptive step for Projected Gradient Descent

The projected gradient method we just presented used a fixed step size. We can however improve the convergence of this method by using _adaptive steps_. In this section, we will present three adaptive step sizes that we will implement to, hopefully, obtain better performances. Note, however, that those step size were originally designed for unconstrained problem. In practice, they also work quite well in the projected case as we will see later with numerical results.

=== Armijo backtracking line search

The first adaptive step size method we will implement is the _Armijo Line Search_. This method starts with a candidate new iterate and decreases the step size until a condition is satisfied. The algorithm is described as follows:

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Armijo Line Search for PGD],
  frame()[
    *Input:* $w_k, c in (0, 1), rho in (0, 1), alpha_0 > 0$ \
    *Init:* $k := 0$\
    *Output:* Step size $alpha_k$ for PGD

    #v(0.5em)

    *Step 1:* $w^(+)_(k) := P_(Delta)(w_k - alpha_k gradient f(w_k))$\
    *Step 2:* \
    - If $f(w^(+)_k) <= f(w_k) - c alpha_k ||gradient f(w_k)||^2$: *return* $alpha_k$\
    - Else: Set $alpha_(k+1) := rho alpha_k, k := k+1$, Go to *Step 1*
  ],
)

This method has several advantages. First of all, it does not require the Lipschiz constant $L$ of the gradient. Also, it offers the same computational complexity as the original projected gradient descent. Most importantly, it often considerably increases the convergence of this method.

=== Barzilai-Borwein step size

Until now, we never used second order information in the model. We may be tempted to implement the Newton's method instead of a Gradient Descent but it requires computing the inverse of the hessian $Sigma$. In practice, it is expensive and sometimes (likely to be the case here) the hessian is singular. The Barzilai-Borwein method, applied to the Gradient Descent, aims to find a step size $alpha_k$ that approximates the Newton step. Here is the description of this method:


#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Barzilai-Borwein Method (Least-Square version)],
  frame()[
    *Input:* $w_k, w_(k-1)$ \
    *Output:* Step size $alpha_k := (s^(k-1)^top s^(k-1))/(s^(k-1)^top y^(k-1))$\
    where $s^(k-1) := w^(k) - w^(k-1), y^(k-1) := gradient f(w_k) - gradient f(w_(k-1))$
  ],
)

Notice that this step size is not defined for the first iteration when $k = 0$. In this case, we just choose $alpha_0 = 1/L$ to perform a regular gradient descent before using the Barzilai-Borwein step for the following iterations.

=== Exact line search

Both adaptive step method we showed previously were _inexact_ step size. Here we will take a look at the _exact_ step size. Here is how it is derived:

Consider the objective value at the new iterate:

#nonumeq(
  $
    f(x_(k+1)) = f(x_k - alpha_k gradient f(x_k)) := J(alpha_k)
  $,
)

This is a function of the step size and can thus be minimized accordingly. After minimizing, we obtain the following expression:

#nonumeq(
  $
    alpha_k = (gradient f(x_k)^top gradient f(x_k))/(gradient f(x_k)^top Sigma gradient f(x_k))
  $,
)

As said previously, the covariance matrix $Sigma$ may contain zero (or close to zero) eigenvalues. The quadratic form at the denominator can therefore be null in some cases. To avoid that in practice, we will precompute the denominator and we use $alpha_k = 1/L$ if it is smaller than a certain tolerance.

== Projected Gradient Descent with Momentum

Here we are going to slightly modify the previous algorithm by introducing the notion of momentum.
In our previous algorithm, we were not taking advantage of the gradient of the previous iterates, i.e. $gradient f(w_(k-1)) , gradient f(w_(k-2)),...,gradient f(w_(0))$. We could then introduce a momentum variable
#nonumeq($ m_(k+1) = beta m_k + (1 - beta ) gradient f(w_k), #h(2em)$)
with $beta in [0,1]$ and $m_0 = 0$.

The projected momentum iterates becomes for an $w_k in Delta$
#nonumeq(
  $
    w_(k+1) = P_Delta (w_k - gamma m_(k+1))
  $,
)


Momentum increases the influence of recent gradients while gradually vanishing older ones, which often accelerates convergence.
The complete algorithm is described as follows :

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Projected Gradient Descent with Momentum],
  frame()[
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
  ],
)

However, with this algorithm (especially with a fixed $beta$), the rate of convergence has the same order $cal(O)(1\/k)$ as the projected gradient descent in the worst-case scenario.\
A better version of this algorithm in the context of convex and smooth functions is the _Nesterov's Accelerated Gradient Method_, whose algorithm is described below:

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Nesterov's accelerated Gradient Method],
  frame()[
    *Input:* $w_0, L$ \
    *Init:* $w_(-1) = w_0, alpha = 1/L, lambda_0 = beta_0 = 0$\
    *Output:* approximate solution $w_N$

    *for* $k = 0, 1, dots, N - 1$ *do* \
    #h(2em) $y_k <-- w_k + beta_k (w_k - w_(k-1))$ \
    #h(2em) $w_(k+1) <-- y_k - alpha gradient f(y_k)$\
    #h(2em) $lambda_(k+1) <-- (1 + sqrt(1 + 4 lambda^2_k))/(2), quad beta_(k+1) <-- (lambda_k - 1)/(lambda_(k+1))$ \
    *end for*
  ],
)

This gives us this theoretical bound:

#nonumeq(
  $
    f(y_k) - f^* <= (2L ||w_0 - w^star||^2)/((k+1)^2), quad forall k >= 1
  $,
)

So our convergence rate is $cal(O)(1\/k^2)$, which is faster than gradient descent's one. This also implies that the number of iterations is $cal(O)(1\/sqrt(epsilon))$ to reach a precision of $epsilon$ on the objective value.

== Projected Randomized Coordinate Descent

The last optimization method that we will implement for this model is the Projected Randomized Coordinate Descent.
In this algorithm, rather than computing the full gradient, we are going to compute one component of the gradient. We are then going to perform one gradient descent in this direction. A description of the method can be written as follows:\
Considering $w_k in Delta$, the step is defined by :
#nonumeq(
  $
    w_(k+1) = P_Delta (w_k - alpha [gradient f(w_k)]_(i_k)) e_(i_k)
  $,
)
with  $i_k tilde cal(U){1,...,n}$

We directly identify several major issues. The first one occurs when we update our weight $w_(k+1)$. Indeed by proceeding in such way, we only update one component of the vector corresponding to the index $i_k$ but projecting back onto the simplex induce a modification of the entire vector.
To fix it we suggest two variants. The first one is the naive one as we are only going to rebalance our weight by modifying an other random weight such that $sum_i w_i = 1$. The second one is a bit more complex. In this version we are also going to take two random weight and we are going to compute their gradient to direct ourself in the direction inducing the biggest variation in term of our objectif function for these two coordinates. The update is performed in such a way that the sum of the two selected weights is preserved, ensuring that the iterate remains in the simplex.
The second issue is related to the stopping criterion from our algorithm. From the beginning we only considered the stopping criterion $||w_(k+1) - w_k|| < epsilon$. However in the randomized coordinate descent, we may fall on a coordinate that won't move that much after the step. This will cause the algorithm to stop, even if the other coordinates are not optimized yet. Thus, the only stop criterion we will use is by applying a limit on the number of iterations.

= Non-smooth Model

== Model

The second model we will study is a variant of the first one. Its formulation is given by:

$
  min_(w in Delta) f(w) = 1/2 w^top Sigma w - lambda w^top mu + c||w - w_"prev"||_1
$<eq:nonsmooth>

In @eq:nonsmooth, we observe that the objective takes the same form as in @eq:smooth. However, there is an additional non-smooth term. This term extends the previous model by taking transaction costs into account. In realistic scenarios, buying or selling assets implies a cost which is proportional to the amount of traded assets. To represent this in the model, we add a term proportional to the $cal(l)_1$-distance between the current portfolio $w$ and a reference one $w_"prev"$:

#nonumeq(
  $
    c||w - w_"prev"||_1
  $,
)

This essentially means that we do not want to drastically change the current allocation of our portoflio. In fact, the $cal(l)_1$-norm encourages _sparse rebalancing_. The parameter $c$ controls the trading cost. A small $c$ implies that the portfolio can change drastically compared to a reference one and vice-versa.

The main difference with the smooth model is that we can not compute the gradient explicitly. Instead, we will have to use methods that either use a subgradient, or other optimization techniques.

In the next sections, we will present the three methods we will implement on this model, that is: projected subgradient method, proximal gradient descent and interior point method. For those methods, we will decompose the previous model into a smooth part and a non-smooth part which we will denote by:

#nonumeq(
  $
        & g(w) := 1/2 w^top Sigma w - lambda w^top mu, quad h(w) := c||w - w_"prev"||_1 \
    ==> & f(w) = g(w) + h(w)
  $,
)

== Projected Subgradient Method

The subgradient method is simply a generalization of the gradient method for non-differentiable function where we can still get access to a subgradient. Let us first derive an expression for the subgradient:

The only non-smooth part of the objective function is $h$, we will therefore find a subgradient for this part. This function is not differentiable at $w = w_"prev"$. The subgradient is thus described by:

#nonumeq(
  $
    partial h(w) = cases(-&c quad &"if" w < w_"prev", &0 quad &"if" w = w_"prev", &c quad &"if" w > w_"prev")
  $,
)

and the subgradient of $f$ is simply the sum of the gradient of $g$ and the subgradient of $h$.

We can now describe the following algorithm for the Projected Subgradient Method:

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Projected Subgradient Method],
  frame()[
    *Input:* $w_0, alpha_k$ \
    *Output:* approximate solution $w_N$

    *for* $k = 0, 1, dots, N - 1$ *do* \
    #h(2em) $w_(k+1) <-- P_(Delta)(w_k - alpha_k partial f(w_k))$\
    *end for*
  ],
)

@subgradient We will now discuss the step size for this method. If we denote $D := max_(w, v in Delta) ||w - v||_2$ the diameter of the simplex and $M >= ||partial f(w_k)||$ the bound on the subgradient, we obtain the following inequality:

$
  min_(k = 0, dots, T) f(x_i) - f^star <= (D^2 + M^2 sum_(k=1)^(T) alpha_k^2)/(2sum_(k=1)^(T) alpha_k)
$<eq:subgd_step_inequality>

We now want to know for what sequence of $alpha_k$ the righthand side of @eq:subgd_step_inequality is minimized. This term is convex and symmetric in $alpha_1, dots, alpha_T$ so the optimum is reached when all the $alpha_k$ are equal ($alpha_k := alpha$). The right term thus simplifies to:

#nonumeq(
  $
    (D^2 + M^2 T alpha^2)/(2T alpha)
  $,
)

which is minimized when $alpha = (D)/(M sqrt(T))$. With this step size, we now want to determine the value $T(epsilon)$ to obtain a certain precision $epsilon$. We have:

#nonumeq(
  $
    min_(k = 0, dots, T) f(x_i) - f^star <= (M D)/(sqrt(T(epsilon))) <= epsilon
  $,
)

We therefore have that $T(epsilon) >= (M^2 D^2)/(epsilon^2) tilde cal(O)(1\/epsilon^2)$. Plugging that in the step size $alpha$:

#nonumeq(
  $
    alpha <= epsilon/M^2
  $,
)


This step size guarantees us to converge in $cal(O)(1\/epsilon^2)$ iterations to obtain an $epsilon$-accuracy solution. However, this convergence rate is quite bad. Even though this is the best we can theoretically do, we may want to choose a diminishing step size to, hopefully, obtain better performances in practice. Additionally to the constant step size we just presented, we will therefore also implement the following diminishing step size:

#nonumeq(
  $
    alpha_k <= epsilon/(M^2 sqrt(k))
  $,
)

== Proximal Gradient Descent

Instead of having to rely on the subgradient of $h$ in our algorithm, we can use the _proximal gradient method_. We define the proximal operator for $h$ given a step size $t$ by:

#nonumeq(
  $
    "prox"_(t, h)(x) = arg min_(z in RR^n) 1/(2t) ||x - z||_2^2 + h(z)
  $,
)

The proximal gradient method is then given by (given $w_0$):

#nonumeq(
  $
    w_k = "prox"_(t_k, h)(w_(k-1) - t_k gradient g(x^(k-1))), quad k = 1, 2, dots
  $,
)

This method intuitively means:

- First term in the $"prox"$ operator: find a point $z$ that stays close to the gradient update of $g$
- Second term in the $"prox"$ operator: Also, try to make $h$ small

Even though this looks like we just reformulated the optimization problem, in some special cases the proximal operator has a closed-form expression. In our case, we have to obtain an expression for the proximal operator of the $cal(l)_1$-norm. It can be proven that, for a function $phi(x) = c||x||_1$, the proximal operator is given by:

#nonumeq(
  $
    ["prox"_(t, phi)(x)]_i = cases(
      x_i - c t quad &"if" x_i > c t,
      0 quad & "if" -c t <= x_i <= c t,
      x_i + c t quad & "if" x_i < -c t
    )
  $,
)

In our case, we have $h(w) = phi(w - w_"prev")$. It is easy to adapt the previous formula to this case\
In the unconstrained case, given a fixed step size $t <= 1/L$ (where $L$ is the smoothness constant of $g$), we have the following convergence result:

#nonumeq(
  $
    f(w_k) - f^star <= (||w_0 - w^star||_2^2)/(2k t)
  $,
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
  $,
)

which thus gives us a convergence rate of $cal(O)(1\/k^2)$ or $cal(O)(1\/sqrt(epsilon))$.

== Long-Step Path-Following Interior-Point method

The _Long-Step Path-Following Interior-Point method_ is a second-order method for solving convex optimization problems. The idea is to include the inequality constraints  in the objective with barrier functions (often logarithmic barriers) and also introduce a _barrier parameter_ which we will denote by $t$ that multiplies the original objective function. We then solve this problem to obtain a solution $x^(star)(t)$ which depends on the barrier parameter. The path ${x^(star)(t) : t > 0}$ is called the _central path_ and as $t->infinity$, we have that $x^(star)(t) -> x^*$. Given a generic convex barrier problem, the long-step variant of IPM is described as follows:

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Long-Step Path-Following IPM],
  frame()[
    *Input:* $x_0, t_0, tau in (0,1), theta in (0, 1)$ \
    *Output:* approximate solution of $x^*$

    *while* $nu\/t_k > epsilon$ *do*\

    #h(2em) $t_(k+1) <-- t_k \/ (1-theta)$\
    #h(2em) *do* _Damped Newton-Steps_ *while* $delta_(t_(k+1))(x_(k+1)) > tau$\
    #h(2em) $k <- k+1$\
    *end while*
  ],
)<algo:ipm>

In @algo:ipm, $tau$ represents the target accuracy within each iteration, $theta$ represents the scaling factor of the barrier parameter and $nu$ is the self-concordant parameter of the barrier function. In the case of logarithmic barriers, this parameter is equal to the number of inequality constraints $m$. The algorithm also uses the _local norm_ $delta_(t)(x)$ which is defined by:

#nonumeq(
  $
    delta_(t)(x) = (gradient f_(t)(x)^top [gradient^2 f_(t)(x)]^(-1) gradient f_(t)(x))^(1\/2)
  $,
)

Let's now try to adapt this method in the case of the non-smooth problem @eq:nonsmooth:

The original problem is described as:

#nonumeq(
  $
       min_(w in RR^n) f(w) & = 1/2 w^top Sigma w - lambda w^top mu + c||w - w_"prev"||_1 \
    "s.t." quad bb(1)^top w & = 1 \
                        w_i & >= 0 quad forall i = 1, dots, n
  $,
)

Before transforming this formulation into a barrier problem, we need to get rid of the absolute values in the objective. To do that, we will introduce slack variables.

Let $u_i, v_i >= 0$ such that $|w_i - w_("prev", i)| = u_i + v_i$ and $w_i - w_("prev", i) = u_i - v_i$. The problem now becomes:

#nonumeq(
  $
    min_(x = (w, u, v) in RR^(3n)) & tilde(f)(x) := 1/2 w^top Sigma w - lambda w^top mu + c dot bb(1)^top (u + v) \
                       "s.t." quad & bb(1)^top w = 1 \
                                   & w_i - w_("prev", i) = u_i - v_i quad forall i = 1, dots, n \
                                   & w_i, u_i, v_i >= 0 quad forall i = 1, dots, n
  $,
)

Which gives us a new variable $x$ of dimension $3n$, $m = 3n$ inequalities and $n+1$ equalities.
Now, we can formulate the barrier problem. In order to do that, we will use logarithmic barriers to introduce the $3n$ inequality constraints in the objective function. We thus obtain:


$
  min_(x in RR^(3n)) & psi_(t)(x) := t tilde(f)(x) - sum_(i=1)^(n) log(w_i) + log(u_i)+ log(u_i) \
         "s.t." quad & bb(1)^top w = 1 \
                     & w_i - w_("prev", i) = u_i - v_i quad forall i = 1, dots, n \
$<eq:barrier_problem>

Note that, because we had $m = 3n$ inequality constraints, the self-concordant parameter of the barrier is $nu = 3n$.

@newton_kkt We now have to derive the damped Newton-Steps for this problem. At each step, we will have to solve the Newton-KKT system to obtain the direction in which we will perform the step. This system is given by:

#nonumeq(
  $
    mat(gradient^2 psi_(t)(x), A^top; A, bold(0)) vec(d_x, beta_"kkt") = -vec(gradient psi_(t)(x), A x - b)
  $,
)

where $d_x$ is the Newton-Step's direction. Here are the expression of the different parts of this system:

#nonumeq($ A = mat(II_n, -II_n, II_n; bb(1)^top, bold(0), bold(0)) $)
#nonumeq(
  $
    cases(
      gradient_w psi_(t)(x) = t(Sigma w - lambda mu) - mat(1/w_1, dots, 1/w_n)^top,
      gradient_u psi_(t)(x) = t c bb(1) - mat(1/u_1, dots, 1/u_n)^top,
      gradient_v psi_(t)(x) = t c bb(1) - mat(1/v_1, dots, 1/v_n)^top,
    )
    => gradient psi_(t)(x) = vec(gradient_w psi_(t)(x), gradient_u psi_(t)(x), gradient_v psi_(t)(x))
  $,
)

#nonumeq(
  $
    cases(
      H_w = t Sigma + "diag"{1/w^2_1, dots, 1/w^2_n},
      H_u = "diag"{1/u^2_1, dots, 1/u^2_n},
      H_v = "diag"{1/v^2_1, dots, 1/v^2_n},
    )
    => gradient^2 psi_(t)(x) = "diag"(H_w, H_u, H_v)
  $,
)

To ensure feasibility of the Newton-Step and a strictly decreasing objective value, we have to damp the Newton steps. Choosing a damping of $1/(1 + delta(x))$ satisfies those two properties. The Newton step thus becomes:

#nonumeq(
  $
    x_(k+1) = x_k + 1/(1 + delta(x)) d_x
  $,
)

To compute the local norm $delta(x)$, we do *not* have to inverse the hessian. In fact, because the Newton step direction $d_x$ satisfies at $x_k$:

#nonumeq(
  $
    d_x = -[gradient^2 psi_(t)(x_k)]^(-1) gradient psi_(t)(x_k)
  $,
)

We can simply compute:

#nonumeq(
  $
    delta_(t)(x_k) = (-gradient psi_(t)(x_k)^top d_x)^(1\/2)
  $,
)

= Numerical results
== Smooth Model 

Before diving into our numerical analysis, note that the optimal value of our objective function $f^star$ was precomputed using a solver from the library `cvxpy` both in the smooth case as in the non-smooth case.

=== Projected Gradient descent

The first experiment we performed was to compute the empirical number of iterations to achieve an $epsilon$-precision solution for many value of $epsilon$.

 
#figure(
  image("../figures/Classical_Projected_Gradient_true_iteration_complexity.svg", width: 80%),
  caption: [Number of iterations vs. $epsilon$ for PGD]
)<fig:PGD_iteration_complexity>

As we can see on @fig:PGD_iteration_complexity, the complexity of the PGD method is way below the theoretical one. In fact, it even looks like that the iteration complexity is of order $cal(O)(1\/sqrt(epsilon))$, which shows that this method is performing way better than expected.

Now as we have mentionned previously, we have tried several adaptive step sizes. Here are the results we obtained : 

#figure(
  image("../figures/Classical_Projected_Gradient_step_size_comparison_objectif_value.svg", width: 80%),
  caption: [$f(x_k)$ vs. $k$ for the different adaptive step sizes ($epsilon = 10^(-8)$)]
)<fig:PGD_step_sizes>

From @fig:PGD_step_sizes, the adaptive step size method offering the best performance in terms of convergence is the _Barzilai-Borwein_ adaptive step. Note, however, that it induces a bigger mean time per iteration than the fixed step size version. Despite the time per iteration being bigger, we only required 7 iterations to converge which mean that the total time is still way less for this adaptive step size than for the other. To illustrate this, we registered the time per iteration for each version to study the mean and the variance. Here are the results : 

#figure(
  grid(
    image("../figures/Classical_Projected_Gradient_step_size_comparison_comparison_computational_cost.svg", width: 80%),
    table(
      columns: 4,
      fill: rgb(0, 180, 250, 30),
      stroke: .1pt,
      align: left,
      table.header[][Mean $["ms"]$][Std $["ms"]$][Iterations],
      [Constant step size], [0.2044], [0.4038], [500],
      [Exact Line search], [0.5080], [1.6201], [493],
      [Backtracking Line Search],[0.4163],[0.4932],[500],
      [Barzilai-Borwein Step Size],[0.4286],[0.4950],[7],
    ),
  ),
  caption: [Time per iteration statistics ($epsilon = 10^(-8)$, `max_iter` = 500)]
)<fig:PGD_time_per_iteration>

On @fig:PGD_time_per_iteration, we can further see how the _Barzilai-Borwein_ step clearly improves the convergence of this method.

=== Projected Gradient descent with momentum

For this method, we will compare both versions of the _PGD with Momentum_ presented previously which are the classic one and the Nesterov Accelerate version.

We will first analyze, as before, the number of iterations to reach convergence for different $epsilon$ and compare both methods.

#figure(
  image("../figures/Classical_Projected_Gradient___true_iteration_complexity.svg", width: 80%),
  caption: [Number of iterations vs. $epsilon$ for PGD with Momentum]
)<fig:Momentum_iteration_complexity>

On @fig:Momentum_iteration_complexity, we see that both curves are strictly below their respective theoretical rate. We also notice that the Nesterov's method slightly better than the classic momentum no matter the value of $epsilon$. To further compare those methods, we will also look at the value of the objective function as a function of the iterations.

#figure(
  image("../figures/Comparison_Momentum_Methods_objectif_value.svg", width: 80%),
  caption: [$f(x_k)$ vs. $k$ for Momentum and Nesterov's Momentum ($epsilon = 10^(-8)$)]
)<fig:Momentum_objective_value>

Again, on @fig:Momentum_objective_value we clearly see that the Nesterov's Momentum converges faster than the classic one. 

=== Projected Randomized Coordinate Descent

For this method, we will take a look at the difference between the objective value at step $k$ and the optimal value $f^star$

#figure(
  image("../figures/Projected_Randomized_Coordinate_Descent_iteration_method_comparison_objective_gap.svg", width: 80%),
  caption: [$f(x_k) - f^star$ vs. $k$ ($"max_iter" = 10^4$)]
)<fig:coordinate_value_diff>

On @fig:coordinate_value_diff, we compared both implementations desribed previously. 
We can observe that, even after $10^4$ iterations, none of the two methods reaches a good approximation for the optimal value. This is likely due to the projection step which causes a slow convergence. From this graph, we can conclude that this method is not suited to this problem.

== Comparison of the model :
We will now compare all the methods shown above by plotting the objective value to observe the convergence.

#figure(
  image("../figures/Projected_Methods_Comparison_Best_ones_objectif_value.svg", width: 80%),
  caption: [$f(x_k)$ vs. $k$ for all the methods ($epsilon = 10^(-8)$)]
)<fig:objective_value_all_smooth_methods>

On @fig:objective_value_all_smooth_methods, we see that the Projected Gradient method with the Barzilai-Borwein step surpasses the other methods by far. Then, the Nesterov's Accelerated Gradient Descent beats the last two. Again, we clearly see that the Randomized Coordinate Descent offers a way worse convergence rate compared to the other methods. It is also interesting to compare the elapsed time per iterations. Here is, like before, a graph showing the statistics about these:

#figure(
  grid(
    image(
      "../figures/Projected_Methods_Comparison_Best_ones_comparison_computational_cost.svg", width: 80%),
    table(
      columns: 4,
      fill: rgb(0, 180, 250, 30),
      stroke: .1pt,
      align: left,
      table.header[][Mean [ms]][Std [ms]][Iterations],
      [PGD + Adaptive step], [0.2857], [0.4518], [7],
      [PGD + Momentum], [0.2826], [0.4703], [811],
      [PGD + Nesterov], [0.2531], [0.4449], [799],

      [Randomized CD], [0.2901], [0.5484], [1000],
    ),
  ),
  caption: [Time per iteration statistics ($epsilon = 10^(-8)$, max_iter = $1000$)],
) <fig:time_per_iteration_all_smooth_models>


We see that PGD with Barzilai-Borwein step and the Nesterov's momentum methods offers the best cost-per-iteration values. One may choose between one of those two for this problem, as they share overall similar performances.


== Influence of $lambda$ on the solution
We will now interpet the effect of the parameter $lambda$ on the smooth Markowitz model with our best algorithm (i.e Projected gradient with Barzilai-Borwein adaptive step).
#figure(
  image("../figures/efficient_frontier_smooth.svg", width: 80%),
  caption: [Efficient frontier for the Smooth Markowitz Model]
)

From what we see on these plots, the efficient frontier has a concave curve. We can interpret that as higher return also mean higher risk which directly translate reality.
For initial values of $lambda in {0.1 , 0.5}$, we have a low-risk but also a low return, increasing the $lambda$ directly yields a bigger return but also a bigger risk. Indeed, the bigger the lambda is the more we try to maximise our return. From the Efficient frontier we see that we have a saturation effect from $lambda = 2$ to $lambda = 20$. We also wanted to analyze the convergence speed with respect to $lambda$:

#figure(
  image("../figures/lambda_convergence_gap.svg", width: 80%),
  caption: [$f(x_k) - f*$  vs $k$ for different $lambda$]
)

In terms of convergence, we see that for bigger $lambda$'s, we converge faster. This is likely due to the fact that the second term becomes dominant and induces a steeper objective landscape.


== Non-Smooth model
// In this part of the numerical analysis we have first taken a constant $c=0.01$ and $lambda=0.5$ in order to compare our models. 
In this section, we will do similar numerical analysis, this time for the methods applied to the non-smooth model @eq:nonsmooth. 

=== Projected Subgradient descent
We will start by analyzing the Subgradient method by first plotting the error $||f(x_k) - f^*||$ as a function of the iteration for different constant step sizes using the previously found rule $alpha = epsilon\/M^2$. 

#figure(
  image("../figures/Subgradient_constant_stepsize_comparison_objective_gap.svg", width: 80%),
  caption: [$||f(x_k) -f^*||$ vs $k$ for several Constant Step $epsilon/M^2$]
)<fig:subgradient_objective_value>

As we can see on @fig:subgradient_objective_value, each curve has the same shape and the larger $epsilon$ is, the faster the convergence. 

We also plotted the complexity curve here to verify that we observe numerically a complexity of $cal(O)(epsilon^(-2))$ : 


#figure(
  image("../figures/Subgradient_complexity_vs_epsilon.svg", width: 80%),
  caption: [Number of iterations vs. $epsilon$ for the Projected Subgradient]
)

By looking at the graph, we can confirm that the convergence rate is better than the theoretical one for this method.

Then we also implemented the diminishing step size for the projected subgradient and we plotted the error with the correct objectif value : 
#figure(
  image("../figures/Subgradient_diminishing_stepsize_comparison_objective_gap.svg", width: 80%),
  caption: [$||f(x_k) -f^*||$ vs iteration for several Constant Step $alpha$ (Tol :$10^(-6)$)]
)
We observe that only two of the three methods have converged, the one starting with an initial step $alpha_0 = 0.01$ and the one with $0.001$. Overall, we see that it does not outperform the constant step size version of this algorithm.

=== Proximal gradient descent

For the proximal descent, we will first plot the error on the objective value as a function of the iterations. The reason why we did not choose a step size of $1/L$ is because the method converged in one iteration and it was thus not interesting to plot it.
#figure(
  image("../figures/Proximal_Gradient_stepsize_comparison_objective_gap.svg", width: 80%),
  caption: [$||f(x_k) -f^*||$ vs iteration for several step sizes])

We directly see that for any step size we quickly converge towards the equilibrium which makes it for now the best method we have. 
We also implemented the Fast Proximal gradient which also converge in a single iteration for $t = 1\/L$ so here is the comparison of the two objective function for a constant step size of $0.01$:
#figure(
  image("../figures/Proximal_Gradient_vs_Fast_objectif_value.svg", width: 80%),
  caption: [$f(x_k)$ vs iteration for Fast and Classic Proximal Gradient ]
)

We then see that it converges in less iterations. However, the time per iteration is larger for the accelerated version: 
#figure(
  image("../figures/Proximal_Gradient_vs_Fast_comparison_computational_cost.svg", width: 80%),
  caption: [Time per iteration for Fast and Classic Proximal Gradient]
)


=== Long-step Path-following Interior-Point method 

Finally, we will perform a similar analysis for the Long-step Path-following Interior point method. Here we have first plotted the objective function: 

#figure(
  image("../figures/InteriorPoint_LongStep_objective_value.svg", width: 80%),
  caption: [$f(x_k)$ vs $k$]
)<fig:IPM_objective_value>

On @fig:IPM_objective_value, we can see that it quickly converges to the optimal value. However, even though the convergence rate is impressive, we have to take into account the cost-per-iteration which is quite high for this method:

#figure(
  image("../figures/InteriorPoint_LongStep_time_complexity.svg", width: 80%),
  caption: [Time per iteration (ms)]
  )

Here, we plotted the elapsed time to reach convergence for this method as the number of stocks $n$ increases. We see that the time complexity is about $cal(O)(n^2)$. In practice, this method takes too much time to be considered a suitable method for this problem. 

== Influence of $c$ on the solution :

In this section we will analyze and interpet the influence of the parameter $c$, just like we did for $lambda$ for the previous model. 
As we mentionned previously, the parameter $c$ controls the sensitivity with respect to the transaction costs. A higher $c$ means that we consider larger transaction costs and vice-versa. Here we have plotted the number of assets for which we have a non-null weight for different value of the parameter $c$. 

#figure(
  image("../figures/portfolio_diversification_vs_transaction_cost.svg", width: 80%),
  caption: [Number of non-zero assets for several value of $c$]
)

Here, we see that the higher the transaction cost, the higher the number of selected assets. This is due to the fact that we used a uniform reference portfolio, for which we do not want to get far away when $c$ is large. When $c$ is really small, we are allowed to change the portfolio considerably, which is why the model will try to invest in a single asset that looks promising.

#figure(
  image("../figures/portfolio_turnover_vs_transaction_cost.svg", width: 80%),
  caption: [Turnover vs $c$]
)

On this graph, we also plotted the _turnover_ of the portfolio. This essentially represents how much, in percentage, our portfolio changed with respect to our reference one. Again, we clearly see that this percentage drastically decreases as the transaction cost increases.

= Conclusion

To conclude this project, we have seen a various number of technic in order to optimise the Smooth and Non-smooth version of the Markovitz problem. The two method for the two respective model that we could highlight as the best performing methods were the Projected Gradient method with adaptive step and the Proximal gradient descent. We also analysed the effect of the parameters $lambda$ and $c$ on the models and observed some interesting result, for instance the Efficient frontier with $lambda$ and the optimum asset allocation with $c$. 


#pagebreak()

#underline("Groupe Number") : 9 

#underline("Students") : 
- Guerand Dewell : Contributed to numerical analysis of methods and implementation
- Lucas Ahou : Contributed to de redaction of the report and also to the implementation

#underline("LLM") : Claude wrote the structure of the code in order to segregate correctly the sections



