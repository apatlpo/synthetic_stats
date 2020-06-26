# Bendat and Piersol: notes

Page count based on 3rd edition

- section 1.4.3 (p23): bias vs random error
- section 3.1.1 (p50): standard distributions with probability density functions
- section 3.1.2 (p53): expected values of a function $g(x)$ where $x$ is a random distribution with density distribution $p(x)$ - $E[p(x)] = \int g(x) p(x) dx$
- section 3.2.2 (p61): distribution of the sum of two variables - $p(z)=\int p_1(x) p_2(z-x) dx$
- section 3.3.1 (p65): central limit theorem - "let $x_1(k), ..., x_N(k)$ be $N$ mutually independent random variables whose individual distributions are not specified and may be different. Let $\mu_i$ and $\sigma_i^2$ be the mean value and variance of each random variable $x_i(k)$. Consider the sum random variable $x_(k) = \sum_{i=1,N} a_i x_i(k)$ where $a_i$ are arbitrary fixed constants. The central limit theorem states that under fairly common conditoins the sum random variable $x(k)$ will be normally distributed as $N\rightarrow\infty$ " with mean and variance straightforwardly related to $a_i$ and $x_i$ mean and variances.
- **section 3.4.1 (p73): distribution of envelope and phase for narrow bandwidth data - Rayleigh distribution**
- example 3.7 (p81): distribution of a sine wave with gaussian noise
- **section 4.1 (86): sample values and parameter estimation - sample mean, variance, unbiased/efficient estimators**
- section 4.2 (p89): important distributions - gaussian, chi-square, t-distribution, F distribution
- **section 4.3.3 (p95): distribution of sample mean with unknown variance**
- section 4.3.4 (p95): distribution of ratio of two sample variances
- **section 4.4 (p96): confidence intervals**
- **section 4.5 (p99): hypothesis tests - chi-square goodness of fit test, non-parametric trend test**
- section 4.6 (p108): correlation and regression procedures
- **example 5.3 (p124): autocorrelation function of sum of two processes**
- table 5.1 et 5.2 (p135): autocorrelation / spectrum correspondance
- section 5.2.9: uncertainty principle - if $T_0=\sqrt{\int t^2 y^2(t) dt}$ and $B_0=\sqrt{\int f^2 Y^2(f) df}$ then $T_0B_0\ge1/4\pi$.
- section 5.5 (p170): level crossings and peak values
- **chapter 8: statistical errors in basic estimates:**
	- section 8.1: definitions of errors (normalized random errors, confidence intervals within 1, 2 std)
	- **section 8.2 (p276): mean and square value estimates**. For bandwidth-limited white noise: $2\epsilon [\hat \mu_x] \sim \sqrt{2/BT} \sigma_x/\mu_x$
	- **section 8.4 (p291): correlation function estimates**. For bandwidth-limited gaussian white noise: $Var(\hat{R}_{xy}(\tau)) \sim \frac{1}{2BT}[R_{xx}(0)R_{yy}(0)+R_{xy}^2(\tau)]$
	- table 8.1: summary table of total record length required.
- section 10.5.1: procedure for analyzing individual records
- **section 12.6.4: analysis procedures for single records** 
- **chapter 13: Hilbert transform**