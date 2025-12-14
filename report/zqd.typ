
The _Long-Step Path-Following Interior-Point method_ (which we will denote IPM for simplicity) is a second-order method for solving convex optimization problems. The idea is to include the inequality constraints  in the objective with barrier functions (often logarithmic barriers) and also introduce a _barrier parameter_ which we will denote by $t$ that multiplies the original objective function. We then solve this problem to obtain a solution $x^(star)(t)$ which depends on the barrier parameter. The path ${x^(star)(t) : t > 0}$ is called the _central path_ and as $t->infinity$, we have that $x^(star)(t) -> x^*$. Given a generic convex barrier problem, the long-step variant of IPM is described as follows:

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Long-Step Path-Following IPM],
  frame()[
    *Input:* $x_0, t_0, tau in (0,1), theta in (0, 1)$ \
    *Output:* approximate solution of $x^*$
    
    *while* $nu\/t_k > epsilon$ *do*\

    #h(2em) $t_(k+1) <-- (1-theta)t_k$\
    #h(2em) *do* _Damped Newton-Steps_ *while* $delta_(t_(k+1))(x_(k+1)) > tau$\
    #h(2em) $k <- k+1$\
    *end while*
  ]
)<algo:ipm>

In @algo:ipm, $tau$ represents the target accuracy within each iteration, $theta$ represents the scaling factor of the barrier parameter and $nu$ is the self-concordant parameter of the barrier function. In the case of logarithmic barriers, this parameter is equal to the number of inequality constraints $m$. The algorithm also uses the _local norm_ $delta_(t)(x)$ which is defined by:

#nonumeq(
  $
    delta_(t)(x) = (gradient f_(t)(x)^top [gradient^2 f_(t)(x)]^(-1) gradient f_(t)(x))^(1\/2)
  $
)

Let's now try to adapt this method in the case of the non-smooth problem @eq:nonsmooth:

The original problem is described as:

#nonumeq(
  $
    min_(w in RR^n) f(w) &= 1/2 w^top Sigma w - lambda w^top mu + c||w - w_"prev"||_1\

    "s.t." quad bb(1)^top w &= 1\
                w_i &>= 0 quad forall i = 1, dots, n
  $
)

Before transforming this formulation into a barrier problem, we need to get rid of the absolute values in the objective. To do that, we will introduce slack variables.

Let $u_i, v_i >= 0$ such that $|w_i - w_("prev", i)| = u_i + v_i$ and $w_i - w_("prev", i) = u_i - v_i$. The problem now becomes:

#nonumeq(
  $
    min_(x = (w, u, v) in RR^(3n)) &tilde(f)(x) := 1/2 w^top Sigma w - lambda w^top mu + c dot bb(1)^top (u + v)\

    "s.t." quad &bb(1)^top w = 1\
                &w_i - w_("prev", i) = u_i - v_i quad forall i = 1, dots, n\
                &w_i, u_i, v_i >= 0 quad forall i = 1, dots, n
  $
)

Which gives us a new variable $x$ of dimension $3n$, $m = 3n$ inequalities and $n+1$ equalities.
Now, we can formulate the barrier problem. In order to do that, we will use logarithmic barriers to introduce the $3n$ inequality constraints in the objective function. We thus obtain:


$
  min_(x in RR^(3n)) &psi_(t)(x) := t tilde(f)(x) - sum_(i=1)^(n) log(w_i) + log(u_i)+ log(u_i)\

  "s.t." quad &bb(1)^top w = 1\
              &w_i - w_("prev", i) = u_i - v_i quad forall i = 1, dots, n\
$<eq:barrier_problem>

Note that, because we had $m = 3n$ inequality constraints, the self-concordant parameter of the barrier is $nu = 3n$. 

@newton_kkt We now have to derive the damped Newton-Steps for this problem. At each step, we will have to solve the Newton-KKT system to obtain the direction in which we will perform the step. This system is given by:

#nonumeq(
  $
    mat(gradient^2 psi_(t)(x), A^top; A, bold(0)) vec(d_x, beta_"kkt") = -vec(gradient psi_(t)(x), A x - b)
  $
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
  $
)

#nonumeq(
  $
    cases(
      H_w = t Sigma + "diag"{1/w^2_1, dots, 1/w^2_n},
      H_u = "diag"{1/u^2_1, dots, 1/u^2_n},
      H_v = "diag"{1/v^2_1, dots, 1/v^2_n},
    )
    => gradient^2 psi_(t)(x) = "diag"(H_w, H_u, H_v)
  $
)

To ensure feasibility of the Newton-Step and a strictly decreasing objective value, we have to damp the Newton-Step. Choosing a damping of $1/(1 + delta(x))$ satisfies those two properties. The Newton-Step is thus:

#nonumeq(
  $
    x_(k+1) = x_k + 1/(1 + delta(x)) d_x
  $
)

To compute the local norm $delta(x)$, we do *not* have to inverse the hessian. In fact, because the Newton-Step direction $d_x$ satisfies at $x_k$:

#nonumeq(
  $
    d_x = -[gradient^2 psi_(t)(x_k)]^(-1) gradient psi_(t)(x_k)
  $
)

We can simply compute:

#nonumeq(
  $
    delta_(t)(x_k) = (-gradient psi_(t)(x_k)^top d_x)^(1\/2)
  $
)
