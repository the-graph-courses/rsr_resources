import { useState, useCallback, useMemo, useRef } from "react";
import { createRoot } from "react-dom/client";

/* ── HELPERS ─────────────────────────────────────────────────────────── */
function ols(pts){const n=pts.length;if(n<3)return null;const mx=pts.reduce((a,p)=>a+p.x,0)/n,my=pts.reduce((a,p)=>a+p.y,0)/n;let ssxx=0,ssxy=0;for(const p of pts){ssxx+=(p.x-mx)**2;ssxy+=(p.x-mx)*(p.y-my);}if(!ssxx)return null;const pN=2,b1=ssxy/ssxx,b0=my-b1*mx,fitted=pts.map(p=>b0+b1*p.x),resid=pts.map((p,i)=>p.y-fitted[i]),rss=resid.reduce((a,r)=>a+r*r,0),mse=rss/(n-pN),rmse=Math.sqrt(Math.max(mse,0)),safeRmse=Math.max(rmse,1e-10),hat=pts.map(p=>1/n+(p.x-mx)**2/ssxx),stdR=resid.map((r,i)=>r/(safeRmse*Math.sqrt(Math.max(1-hat[i],1e-10)))),studentRes=resid.map((r,i)=>{const denomDf=Math.max(n-pN-1,1),looVar=Math.max((rss-(r*r)/Math.max(1-hat[i],1e-10))/denomDf,1e-10);return r/(Math.sqrt(looVar)*Math.sqrt(Math.max(1-hat[i],1e-10)));}),cooks=stdR.map((s,i)=>(s*s*hat[i])/(pN*Math.max(1-hat[i],1e-10)));return{b0,b1,fitted,residuals:resid,stdRes:stdR,studentRes,hat,cooks,mse,rmse,mx,n};}
function qnorm(p){if(p<=0)return -Infinity;if(p>=1)return Infinity;if(p===.5)return 0;const f=p<.5,pp=f?p:1-p,t=Math.sqrt(-2*Math.log(pp));let z=t-(2.515517+.802853*t+.010328*t*t)/(1+1.432788*t+.189269*t*t+.001308*t*t*t);return f?-z:z;}
function qqPts(stdR){const idx=stdR.map((r,i)=>({v:r,i}));idx.sort((a,b)=>a.v-b.v);return idx.map((it,i)=>({th:qnorm((i+.5)/idx.length),sa:it.v,oi:it.i}));}
function qqWorm(stdR){
  const qq=qqPts(stdR),n=stdR.length;
  const m=stdR.reduce((a,v)=>a+v,0)/n;
  const sd=Math.sqrt(stdR.reduce((a,v)=>a+(v-m)**2,0)/Math.max(n-1,1));
  return qq.map((q,i)=>{
    const pi=(i+0.5)/n;
    const phi=Math.exp(-q.th*q.th/2)/Math.sqrt(2*Math.PI);
    const seOrd=phi>1e-9?Math.sqrt(pi*(1-pi)/(n*phi*phi)):Infinity;
    const half=Math.min(1.96*seOrd,Math.max(2,3*sd));
    return{...q,dev:q.sa-(m+sd*q.th),lo:-half,hi:half};
  });
}
function loess(xV,yV,nP=60,span=0.75,degree=2){
  const n=xV.length;if(n<4)return[];
  const idx=xV.map((_,i)=>i).sort((a,b)=>xV[a]-xV[b]);
  const xs=idx.map(i=>xV[i]),ys=idx.map(i=>yV[i]);
  const xMn=xs[0],xMx=xs[n-1];
  if(xMx-xMn<1e-12)return[];
  const k=Math.max(degree+2,Math.min(n,Math.ceil(span*n)));
  const minBW=Math.max((xMx-xMn)/(n*8),1e-9);
  const fitAt=(xp)=>{
    const ds=new Array(n);for(let j=0;j<n;j++)ds[j]=Math.abs(xs[j]-xp);
    const sortedDs=[...ds].sort((a,b)=>a-b);
    const bw=Math.max(sortedDs[Math.min(k-1,n-1)],minBW);
    let S0=0,S1=0,S2=0,S3=0,S4=0,T0=0,T1=0,T2=0,sumW2=0;
    const ws=new Array(n).fill(0);
    for(let j=0;j<n;j++){const u=ds[j]/bw;if(u>=1)continue;const w=(1-u**3)**3;ws[j]=w;const xc=xs[j]-xp,xc2=xc*xc;S0+=w;S1+=w*xc;S2+=w*xc2;S3+=w*xc2*xc;S4+=w*xc2*xc2;T0+=w*ys[j];T1+=w*xc*ys[j];T2+=w*xc2*ys[j];sumW2+=w*w;}
    if(S0<1e-9)return null;
    let yhat=null;
    if(degree>=2){
      const detM=S0*(S2*S4-S3*S3)-S1*(S1*S4-S3*S2)+S2*(S1*S3-S2*S2);
      if(Math.abs(detM)>1e-10){const detA=T0*(S2*S4-S3*S3)-S1*(T1*S4-S3*T2)+S2*(T1*S3-S2*T2);yhat=detA/detM;}
    }
    if(yhat==null&&degree>=1){const denom=S0*S2-S1*S1;if(Math.abs(denom)>1e-12)yhat=(T0*S2-T1*S1)/denom;}
    if(yhat==null)yhat=T0/S0;
    return{yhat,ws,sw:S0,sumW2};
  };
  const fitsAtData=xs.map(x=>{const f=fitAt(x);return f?f.yhat:null;});
  const out=[];
  for(let i=0;i<nP;i++){
    const xp=xMn+(xMx-xMn)*i/(nP-1);
    const f=fitAt(xp);if(!f)continue;
    let wSSE=0;
    for(let j=0;j<n;j++){if(f.ws[j]>0&&fitsAtData[j]!=null){const e=ys[j]-fitsAtData[j];wSSE+=f.ws[j]*e*e;}}
    const dfEff=Math.max(f.sw-(degree+1),1);
    const sigma2=Math.max(wSSE/dfEff,0);
    const seFit=Math.sqrt(sigma2*f.sumW2)/f.sw;
    out.push({x:xp,y:f.yhat,lo:f.yhat-1.96*seFit,hi:f.yhat+1.96*seFit});
  }
  return out;
}

/* ── DATASETS ────────────────────────────────────────────────────────── */
const DS={
lin_good:{label:"Exercise & Resting Heart Rate",desc:"Weekly exercise hours vs. resting heart rate (bpm) for 30 adults in a wellness program. A clean, linear negative relationship.",points:(()=>{const xs=[1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,2,3,4.5,6,7,8,9,1.5,5,8.5];const ns=[2,-1,3,-2,1,-3,2,-1,4,-2,1,-3,2,-4,3,-1,2,-2,1,-3,0,2,-1,3,-2,1,-1,2,-3,1];return xs.map((x,i)=>({x,y:+(82-2.2*x+ns[i]*1.8).toFixed(1)}));})()},
lin_border:{label:"Age & Systolic Blood Pressure",desc:"Age vs. systolic BP from a community screening. BP rises with age, possibly accelerating slightly. Subtle enough to debate.",points:(()=>{const xs=[25,28,31,34,37,40,43,46,49,52,55,58,61,64,67,70,27,33,39,45,51,57,63,69,30,36,42,48,54,60];const ns=[3,-2,4,-3,1,-4,2,-1,5,-3,2,-5,3,-1,4,-2,5,-3,2,-4,3,-2,1,-3,4,-2,2,-3,1,-4];return xs.map((x,i)=>({x:+x.toFixed(0),y:+(100+.45*x+.003*x*x+ns[i]*2.5).toFixed(0)}));})()},
lin_bad:{label:"Drug Dosage & Pain Relief",desc:"Analgesic dosage (mg) vs. pain relief score. Relief rises fast at low doses but clearly plateaus. A straight line badly misses this curve.",points:(()=>{const xs=[2,4,6,8,10,14,18,22,26,30,35,40,45,50,55,60,65,70,3,9,16,24,32,42,52,62,7,20,38,58];const ns=[1,-1,1.5,-1,.5,-1.5,1,-.5,1.5,-1,.5,-1,1,-1.5,.5,-1,1.5,-.5,1,-1,.5,1,-1.5,.5,-1,1,-.5,1.5,-1,.5];return xs.map((x,i)=>({x,y:+(18*Math.sqrt(x)+ns[i]*2.5).toFixed(1)}));})()},
hom_good:{label:"Exercise Duration & Calories Burned",desc:"Minutes of exercise vs. calories burned. The spread stays consistent from short to long workouts.",points:(()=>{const xs=[15,20,22,25,28,30,33,35,37,40,42,45,48,50,52,55,58,60,63,65,18,27,34,39,44,49,54,57,62,36];const ns=[12,-8,15,-10,6,-14,9,-7,16,-11,5,-13,10,-6,14,-9,7,-12,11,-8,13,-10,8,-15,6,-11,14,-7,9,-12];return xs.map((x,i)=>({x,y:+(50+5.8*x+ns[i]).toFixed(0)}));})()},
hom_border:{label:"Education & Health Literacy",desc:"Years of education vs. health literacy score. Scores widen slightly at higher education levels, but the pattern is subtle.",points:(()=>{const xs=[8,9,10,10,11,11,12,12,12,13,13,14,14,14,15,15,16,16,16,17,17,18,18,19,20,9,11,13,15,17];const ns=[1.5,-1,2,-1.5,2.5,-2,1.5,-2.5,3,-1,2.5,-2,3,-3,2,-3.5,3.5,-2.5,4,-3,3.5,-4,3,-4.5,5,1,-2,2.5,-3,3.5];return xs.map((x,i)=>({x,y:+(20+4*x+ns[i]*(1+(x-14)*.04)).toFixed(1)}));})()},
hom_bad:{label:"Income & Medical Spending",desc:"Household income ($K) vs. annual out-of-pocket medical spending. Wealthier households show much more variable spending, creating a clear fan shape.",points:(()=>{const xs=[20,25,28,32,35,38,42,45,48,52,55,58,62,65,68,72,75,80,85,90,95,100,110,120,22,40,56,70,88,105];const ns=[.2,-.1,.3,-.4,.5,-.6,.8,-.7,1,-.9,1.2,-1.1,1.4,-1.3,1.5,-1.4,1.7,-1.6,1.9,-1.8,2.1,-2,2.4,-2.2,.15,-.5,.9,-1.2,1.8,-2.3];return xs.map((x,i)=>({x,y:+(200+12*x+ns[i]*x*.8).toFixed(0)}));})()},
inf_good:{label:"BMI & Total Cholesterol",desc:"BMI vs. total cholesterol (mg/dL) from a routine health screening. No point exerts outsized influence.",points:(()=>{const xs=[19,20,21,21.5,22,22.5,23,23.5,24,24.5,25,25.5,26,26.5,27,27.5,28,28.5,29,29.5,30,30.5,31,31.5,32,20.5,23.5,26,28,30];const ns=[5,-3,7,-4,2,-6,4,-2,8,-5,3,-7,5,-3,6,-4,7,-5,3,-6,8,-4,5,-7,6,4,-5,3,-4,7];return xs.map((x,i)=>({x,y:+(120+4.5*x+ns[i]*2).toFixed(0)}));})()},
inf_border:{label:"Unusually Healthy Elder",desc:"Age vs. systolic BP, with one 78-year-old whose BP is lower than expected. High leverage, but roughly on trend.",points:(()=>{const xs=[30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,33,39,45,51,57,63,35,43,55,61];const ns=[3,-2,4,-3,1,-4,2,-1,5,-3,2,-4,3,-2,4,-1,5,-3,2,-4,3,-2,1,-3,4,-2,2,-3,1,-4];const p=xs.map((x,i)=>({x,y:+(90+.8*x+ns[i]*2).toFixed(0)}));p.push({x:78,y:140});return p;})()},
inf_bad:{label:"Data Entry Error in BMI Study",desc:"Same cholesterol screening, but one record has BMI=42 with cholesterol=130. Likely a data entry error. Cook's distance flags it immediately.",points:(()=>{const xs=[19,20,21,22,23,23.5,24,24.5,25,25.5,26,26.5,27,27.5,28,28.5,29,29.5,30,30.5,31,20.5,23,25.5,27,29,22,24.5,26.5,28.5];const ns=[5,-3,7,-4,2,-6,4,-2,8,-5,3,-7,5,-3,6,-4,7,-5,3,-6,8,4,-5,3,-4,7,-3,6,-5,4];const p=xs.map((x,i)=>({x,y:+(120+4.5*x+ns[i]*2).toFixed(0)}));p.push({x:42,y:130});return p;})()},
norm_good:{label:"Height & Lung Capacity",desc:"Height (cm) vs. lung capacity (L) for 30 adults. Prediction errors distribute symmetrically.",points:(()=>{const xs=[155,157,159,160,162,163,165,166,167,168,170,171,172,173,174,175,177,178,180,182,184,186,188,190,158,164,169,176,181,185];const ns=[.15,-.1,.08,-.2,.12,-.05,.18,-.14,.03,-.17,.11,-.08,.16,-.12,.06,-.19,.13,-.07,.1,-.15,.09,-.11,.17,-.13,.04,-.16,.14,-.09,.07,-.06];return xs.map((x,i)=>({x,y:+(-3+.042*x+ns[i]).toFixed(2)}));})()},
norm_border:{label:"Hours of Sleep & Sick Days",desc:"Average nightly sleep vs. sick days per year. Most people follow the trend, with only a mild tail wiggle.",points:(()=>{const xs=[5,5.5,5.5,6,6,6.5,6.5,6.5,7,7,7,7,7.5,7.5,7.5,7.5,8,8,8,8,8.5,8.5,8.5,9,9,5,6,7,8,8.5];const ns=[-1.1,.2,-.3,.45,-.1,.3,-.2,.5,-.15,.1,-.3,.25,-.4,.15,.05,-.25,.35,-.1,.2,-.35,1.0,.3,-.2,-.15,-.4,.9,.4,-.05,.15,-.8];return xs.map((x,i)=>({x,y:+(18-1.5*x+ns[i]*1.1).toFixed(1)}));})()},
norm_bad:{label:"Poverty Rate & ER Visits",desc:"Neighborhood poverty rate (%) vs. ER visits per 1,000. Most follow the trend, but several neighborhoods have extremely high visit rates. Clearly right-skewed residuals.",points:(()=>{const xs=[5,7,8,10,12,13,15,16,18,19,20,22,23,25,27,28,30,32,34,35,37,38,40,42,6,14,21,29,36,41];const sk=[.1,.05,.15,.05,3.4,.1,.2,.08,4.2,.08,.12,.06,5.1,.12,.2,.04,3.2,.15,.08,.06,6.2,.08,.16,.02,.1,.1,4.8,.05,.12,6.7];return xs.map((x,i)=>({x,y:+(80+3.5*x+sk[i]*22-6).toFixed(0)}));})()},
};

/* ── MCQ DATA ────────────────────────────────────────────────────────── */
const MCQ={
sample:[
{q:"Both panels show the same true relationship; a single new red point has been added to each. In which panel did the new point most reshape the fitted line?",v:"sample",opts:["The n = 8 panel","The n = 80 panel"],ans:0,explain:"With only 8 observations, one unusual point can noticeably rotate the fitted line. With 80 observations, the same kind of point has much less leverage over the whole estimate."},
{q:"The two intervals shown are centred on the same slope estimate. Which sample size produced the wider confidence interval?",v:"sample_ci",opts:["n = 200","n = 15"],ans:1,explain:"The point estimate is the same, but the smaller sample has more uncertainty, so its confidence interval is wider."},
{q:"A model is fit on n = 8 observations. The residuals-vs-fitted plot looks \u201Cclean\u201D \u2014 points scattered fairly evenly around zero, with no obvious curve, fan, or extreme outlier. What is the right takeaway?",v:"sample_residual_ok",opts:["With only 8 residuals, even a flat-looking plot is weak evidence \u2014 patterns that violate the assumptions can stay invisible at this sample size","The residuals show no systematic problem, so the modeling assumptions appear satisfied and the usual coefficient estimates and confidence intervals can be reported with the standard interpretation"],ans:0,explain:"Diagnostic plots are weak with very small samples because there are too few residuals to reveal patterns reliably. A flat-looking plot is not the same as evidence that the assumption holds."},
{q:"Which statement about sample size is most accurate?",opts:["Never fit regression with fewer than 30 observations","Small samples can be fit, but estimates and diagnostics are fragile","Sample size only matters in multiple regression"],ans:1,explain:"There is no magic cutoff. The key teaching point is that small samples make slopes, standard errors, and diagnostic plots less stable."},
],
uncorrelated:[
{q:"The picture groups dots into three clinics. What is the concern?",v:"cluster",opts:["Within-clinic residual variance exceeds between-clinic variance, biasing the slope estimate","Residuals from the same clinic look alike, so observations are not fully independent"],ans:1,explain:"Patients in the same clinic share staff, systems, and local conditions, so their residuals tend to look alike. The within-vs-between variance phrasing in option A sounds technical, but cluster correlation primarily inflates the standard errors \u2014 it does not by itself bias the slope."},
{q:"In the picture, residuals plotted in collection order show long runs of positive, then negative, then positive values. What does this suggest?",v:"time",opts:["Normal random variation","Autocorrelation"],ans:1,explain:"Random residuals should bounce around zero without long runs. A wave or run in collection order suggests nearby observations have related errors."},
{q:"Which of these assumptions is hardest to diagnose from the four standard residual plots alone?",ev:"four_diag",opts:["Linearity","Normality","Independence"],ans:2,explain:"The standard diagnostic plots can look clean even when observations are not independent. Independence depends heavily on how the data were collected."},
],
multicollinearity:[
{q:"A researcher fits the model:\u00A0\u00A0blood_pressure = \u03B2\u2080 + \u03B2\u2081\u00B7age + \u03B5. Why can multicollinearity not occur in this model?",opts:["There is only one predictor","The predictor is normally distributed"],ans:0,explain:"Multicollinearity means predictors overlap with other predictors. A simple regression has only one predictor, so there is no second predictor for it to be collinear with."},
{q:"A model is fit as:\u00A0\u00A0health_score = \u03B2\u2080 + \u03B2\u2081\u00B7education + \u03B2\u2082\u00B7income + \u03B5. Education and income turn out to be very strongly correlated with each other in the data. What does this imply for the coefficient estimates?",opts:["The two predictors carry overlapping information, making it hard to estimate their separate effects","The predictors will jointly produce non-normal residuals"],ans:0,explain:"The overlap means the predictors move together, so the model has trouble deciding which predictor deserves credit for the shared part of the signal. The coefficient standard errors inflate."},
{q:"Education and income are strongly correlated in your data and you include both as predictors. What should you expect compared with a model that uses only one of them?",opts:["The standard errors of the individual coefficients will tend to be larger, even though the total R\u00B2 may change only modestly","Including both predictors will roughly double the model's explanatory power, since each predictor contributes independent information"],ans:0,explain:"Highly correlated predictors leave the model unable to separate their individual contributions, which inflates the standard errors. The shared information is largely already in either predictor alone, so R\u00B2 often does not increase by much."},
],
shape:[
{q:"The dots clearly trace a curve, yet a straight regression line cuts across them. What does that mean for a single reported slope?",v:"curve",opts:["The slope misrepresents the effect because it is not constant across x","Nothing; the slope is still valid"],ans:0,explain:"A straight line forces one average slope onto a relationship whose slope changes across x."},
{q:"In this residuals-vs-fitted plot, what is the most reassuring feature for the linearity assumption?",v:"resid_good",opts:["The smoother stays close to zero across the full range of fitted values","The positive and negative residuals balance out, summing to roughly zero"],ans:0,explain:"A flat smoother near zero means the linear form is not leaving a systematic pattern. The fact that residuals sum to zero is automatic for OLS and tells you nothing about whether the relationship is linear \u2014 a clear U-shape can sum to zero just as easily."},
{q:"A transformation can sometimes repair curvature. The explanation plot below shows the same data plotted as y vs x (curved) and as log(y) vs x (much straighter). In one or two sentences, explain why a log transform of y can turn a curved relationship into a roughly linear one.",ev:"log_fix",prose:"A log transformation compresses large values of y much more than small ones. When y grows multiplicatively with x \u2014 for example, exponential growth or a saturating decay \u2014 taking log(y) linearises the relationship, and a straight line becomes a much better summary. The same regression machinery applies after transforming, but the slope is now interpreted on the log scale (e.g. as a percentage change in y per unit of x)."},
],
homogeneity:[
{q:"Three residual-vs-fitted patterns are shown. Which one most clearly indicates heteroscedasticity?",v:"het_pick",opts:["The middle panel","The left panel","The right panel"],ans:0,explain:"Heteroscedasticity means the spread of residuals changes with the fitted value. The middle panel shows residuals widening from left to right \u2014 the classic fan. The left panel has roughly constant spread (homoscedastic), and the right panel shows a U-shape, which is a linearity issue rather than a variance one."},
{q:"In this scale-location plot, what does the upward-sloping smoother indicate?",v:"scale_loc",opts:["Residual spread is increasing with fitted values","The residuals are becoming more normal"],ans:0,explain:"The y-axis is error magnitude (\u221A|standardised residual|). A rising smoother means typical residual size grows as fitted values increase, which is heteroscedasticity."},
{q:"When residual variance is clearly non-constant, why are the usual confidence intervals unreliable?",opts:["The standard formula assumes constant variance, so the resulting interval can be too narrow or too wide","The slope estimate itself becomes biased, so any interval centred on it lands in the wrong place"],ans:0,explain:"OLS standard errors are derived under the assumption of constant variance, and heteroscedasticity distorts that calculation. Robust (HC) standard errors recompute it without that assumption. The slope itself can still be unbiased."},
{q:"Why might income vs. medical spending look like a fan-shaped scatter?",v:"income_fan",opts:["Higher-income households have more discretionary spending, so individual choices add more variability at the top","Higher-income households spend more on average, so the regression line is steeper at the high end"],ans:0,explain:"The fan is about variance, not the mean. At higher incomes, spending depends more on personal choice, which inflates the spread. A steeper average slope at the top would be a linearity issue, not heteroscedasticity."},
],
influential:[
{q:"Three panels show different points marked in red. Which point is most influential on the fitted line?",v:"outlier_types",opts:["The off-trend point with high leverage (right panel)","The low-leverage outlier (left panel)","The high-leverage point on the line (middle panel)"],ans:0,explain:"Influence usually requires both ingredients: unusual x-position and a sizable residual. The right-panel point has both."},
{q:"The red point in this leverage-vs-residual plot sits near the Cook's distance contour. Why is that point flagged as concerning?",v:"lev_resid",opts:["It combines an unusual X value with a sizeable residual, so removing it could meaningfully shift the fitted line","It has the largest leverage of any point, which alone is enough to make it concerning regardless of where it sits relative to the line"],ans:0,explain:"Influence is the combination of leverage and residual size. A point with high leverage but a tiny residual sits roughly on the line and barely moves the fit \u2014 leverage on its own is not enough. The Cook's-distance contour captures both ingredients together."},
{q:"Cook's distance combines two ingredients. Which pair is correct?",v:"cooks_combine",opts:["Leverage (unusual X) and residual size (off the line)","The number of predictors and the residual standard error"],ans:0,explain:"A point with high leverage but a tiny residual may not matter much. A point with both high leverage and a large residual can strongly move the fit."},
{q:"The figure shows the same data fitted with and without the red point. What is the danger this illustrates for the conclusions you would report?",v:"slope_shift",opts:["The reported relationship may rest heavily on a single observation rather than on the overall pattern","Removing flagged points always tightens the confidence interval and improves R\u00B2"],ans:0,explain:"The danger is not that the observation exists \u2014 it is that the scientific story can flip when one point is excluded. R\u00B2 and CI behaviour after removal depends on the data; it is not guaranteed in either direction."},
],
normality:[
{q:"In this detrended Q-Q plot, the points trail away from the zero line at both ends, while the middle stays close to it. What is this most likely a sign of?",v:"qq_heavy_tails",opts:["Heavier tails than a normal distribution","Approximately normal residuals","A non-linear x-y relationship"],ans:0,explain:"In a detrended Q-Q plot, the diagonal of the regular Q-Q has been flattened to a horizontal zero line. Points pulling away at both ends are the signature of heavier-than-normal tails (often extreme outliers). The middle staying close to zero rules out a strong overall skew, and a Q-Q plot speaks to the residual distribution, not the x-y relationship itself."},
{q:"In this detrended Q-Q plot, the points scatter randomly around the horizontal zero line, mostly within the shaded confidence band. What does that suggest about the residuals?",v:"worm",opts:["Roughly consistent with a normal distribution","Clearly skewed","Heavy tails"],ans:0,explain:"In the detrended version, the diagonal of a regular Q-Q has been flattened to horizontal. Random bouncing around zero, with most points inside the confidence band, is the signature of approximate normality."},
{q:"For which sample size does the normality assumption matter most for the usual p-values and confidence intervals?",ev:"clt_safety",opts:["The small (n = 10) sample","The large (n = 200) sample, because of the CLT"],ans:0,explain:"With small samples, p-values and confidence intervals lean more heavily on the normality assumption. With large samples, the CLT often helps, but it is not a magic fix for extreme outliers, dependence, or the wrong model shape."},
{q:"What does non-normality of residuals primarily affect?",ev:"pred_vs_inf",opts:["The reported confidence intervals and p-values","The fitted predictions, but not the standard errors"],ans:0,explain:"The fitted line can still be useful for prediction. The bigger concern is whether the usual uncertainty statements are trustworthy."},
],
exogeneity:[
{q:"A study reports that more physical therapy visits predict higher pain scores. The diagram suggests one reason this slope cannot be read as 'therapy causes more pain.' What is the issue?",v:"confound",opts:["An omitted common cause (injury severity) drives both variables","The sample size is too small"],ans:0,explain:"Injury severity can increase both the number of therapy visits and the pain score. The therapy-visits slope is therefore not a clean causal effect on pain."},
{q:"All four standard residual plots look clean. Does this prove that exogeneity holds?",opts:["No; exogeneity is about causal structure, not about residual shapes","Yes; clean residual plots together with a normal Q-Q rule out omitted-variable bias"],ans:0,explain:"Clean diagnostic plots cannot prove that an omitted confounder is absent. Exogeneity is a design and causal reasoning issue, not a residual-shape one."},
{q:"In the age-vs-cars diagram, what omitted background factor could create a misleading link?",v:"age_cars",opts:["Life stage and wealth","Random measurement error in the recorded age"],ans:0,explain:"Age may appear related to cars owned because life stage and wealth affect car ownership and are also related to age. The causal story is not simply 'age causes cars.'"},
{q:"What is the best way to assess whether exogeneity is plausible?",opts:["Substantive domain knowledge to identify potential confounders","Inspecting all four standard residual plots together"],ans:0,explain:"No residual plot can certify exogeneity. You need theory, design knowledge, and a careful search for plausible omitted causes."},
],
};

/* ── SUNSHINE CONFIG ─────────────────────────────────────────────────── */
const SUNSHINE=[
{key:"sample",letter:"S",label:"Sample Size",type:"info",color:"#9B6B2F",colorSoft:"#F5E6D0",summary:"Do you have enough observations for stable estimates?"},
{key:"uncorrelated",letter:"U",label:"Uncorrelated Errors",type:"info",color:"#2E86AB",colorSoft:"#D0EAF5",summary:"Are residuals independent of each other?"},
{key:"multicollinearity",letter:"N",label:"No Multicollinearity",type:"info",color:"#6B6B6B",colorSoft:"#ECECEC",summary:"Are predictors too correlated? (Multiple regression only)"},
{key:"shape",letter:"S",label:"Shape",labelParen:"(Linearity)",type:"diagnostic",diagKey:"linearity",color:"#2B6CB0",colorSoft:"#D4E3F5",summary:"Is the relationship actually a straight line?",
  explanation:"For one predictor, you can sometimes see non-linearity in the scatter plot. The fitted-vs-residuals plot makes curvature much more obvious. If the smooth line stays near zero, linearity holds. A U-shape means the relationship is curved.",
  plotGuide:"**X-axis: fitted values.** The model's predicted y for each observation.\n**Y-axis: residuals.** Actual minus predicted y, the part of the data the line did not capture.\nThe dashed horizontal line at zero marks perfect predictions. The green smooth line traces the average residual at each fitted value. If it curves, the real relationship is not linear.",
  formalTest:"Ramsey RESET test",whatBreaks:"**Predictions can be systematically off** in some regions of x, and the **slope estimate is biased relative to the true effect**.",
  howToFix:"Try **transforming the predictor** (log, square root), **adding a polynomial term**, or using a **non-linear model**.",
  examples:{good:"lin_good",borderline:"lin_border",bad:"lin_bad"}},
{key:"homogeneity",letter:"H",label:"Homogeneity of Variance",labelParen:"(Homoscedasticity)",type:"diagnostic",diagKey:"homogeneity",color:"#2F855A",colorSoft:"#D4EDDF",summary:"Does the spread of residuals stay constant?",
  explanation:"The variance of prediction errors should be roughly the same at every level of the predictor. When it fans out (heteroscedasticity), confidence intervals and p-values become unreliable.",
  plotGuide:"**X-axis: fitted values.** The model's predicted y for each observation.\n**Y-axis: \u221A|standardized residuals|.** Error magnitude on a z-score-like scale, with the square root compressing the tail so your eye can judge whether the smooth line is flat.\nA flat line means constant spread. An upward slope means variance is growing with the fitted value.",
  formalTest:"Breusch-Pagan test",whatBreaks:"**Standard errors are typically off** (often understated). **Confidence intervals and p-values become unreliable**. If the line is otherwise the right model and there are no omitted-variable problems, the coefficient can still be unbiased, but the usual inference around it cannot be trusted.",
  howToFix:"Use **robust standard errors** (HC3). Or **transform the outcome** (log often stabilizes variance). Or use **weighted least squares**.",
  examples:{good:"hom_good",borderline:"hom_border",bad:"hom_bad"}},
{key:"influential",letter:"I",label:"Influential Points",type:"diagnostic",diagKey:"influential",color:"#C53030",colorSoft:"#FED7D7",summary:"Is any single point pulling the regression line too much?",
  explanation:"An influential observation disproportionately determines where the line goes. It needs to be unusual in two ways: far from the other X values (high leverage) and far from where the line would go without it (large residual).",
  plotGuide:"**X-axis: leverage.** How unusual the observation's predictor value is.\n**Y-axis: standardized residual.** How far the observation sits from the fitted line, on a z-score-like scale.\nThe dashed curve is a Cook's distance contour at the F(0.5, p, n−p) quantile (matching R's performance::check_model). Points beyond the curve combine high leverage with a large residual, the two ingredients of an influential observation.",
  formalTest:"Cook's distance (typical threshold for simple regression: F(0.5, p, n−p) ≈ 0.7), DFFITS, DFBETAS",whatBreaks:"A single observation can **shift the slope substantially**. Your **conclusions may depend on one data point**.",
  howToFix:"**Investigate flagged points**. If they look like data errors, **correct or remove them**. If they are real, **consider reporting results with and without them**.",
  examples:{good:"inf_good",borderline:"inf_border",bad:"inf_bad"}},
{key:"normality",letter:"N",label:"Normality of Residuals",type:"diagnostic",diagKey:"normality",color:"#6B46C1",colorSoft:"#E9D8FD",summary:"Do prediction errors follow a bell curve?",
  explanation:"Normal residuals are mainly important for the usual confidence intervals and p-values, especially with small samples. The fitted line itself does not need perfectly normal residuals to be useful, but strong skew, heavy tails, or extreme outliers can make the uncertainty statements misleading.",
  plotGuide:"**X-axis: theoretical normal quantiles.** Where each residual would sit if the residuals were exactly normal.\n**Y-axis: deviation from the normal line.** How far the observed residual quantile lands from that theoretical position. The diagonal of a regular Q-Q plot has been 'flattened' to a horizontal zero line (this is the detrended Q-Q used by performance::check_model).\nA shaded confidence band gives a tolerance. As long as most points scatter inside the band, mild departures are not a concern. A systematic curve, an S-shape, or many points outside the band suggests the residual distribution is not normal.",
  formalTest:"Shapiro-Wilk test",whatBreaks:"**Confidence intervals and p-values can be unreliable**, especially in small samples. Coefficient estimates are **not biased simply because residuals deviate from normality**. The bigger risk is when 'non-normal' really means extreme outliers or a misspecified model, which can genuinely distort the fit.",
  howToFix:"Try **transforming the outcome** (log, square root). **Investigate extreme values**. With large samples, moderate non-normality is often less serious.",
  examples:{good:"norm_good",borderline:"norm_border",bad:"norm_bad"}},
{key:"exogeneity",letter:"E",label:"Exogeneity",type:"info",color:"#8B6914",colorSoft:"#F0E4C8",summary:"Is the predictor independent of the error term?"},
];

/* ── COLORS ──────────────────────────────────────────────────────────── */
const C={bg:"#fafaf9",card:"#FFFFFF",panel:"#fbfbfa",border:"#e7e5e4",text:"#1c1917",sub:"#57534e",muted:"#78716c",grid:"#f1f5f4",teal:"#006D77",tealDark:"#005A63",tealLight:"#83C5BE",tealBg:"#e8f6f5",gold:"#FFD166",ref:"#C83B3B",smooth:"#006D77",dash:"#334155",dot:"#006D77",dotS:"#005A63",hi:"#C83B3B",hiG:"#C83B3B22",shadow:"0 10px 28px rgba(35,48,56,0.08)"};

/* ── CHART ───────────────────────────────────────────────────────────── */
const PAD={top:22,right:16,bottom:42,left:50};
function sc(v,sz){if(!v.length)return{fn:()=>sz/2,min:0,max:1,ticks:[]};let mn=Math.min(...v),mx=Math.max(...v);if(mx-mn<1e-9){mn-=1;mx+=1;}const p=(mx-mn)*.08,lo=mn-p,hi=mx+p,fn=vv=>((vv-lo)/(hi-lo))*sz;const rg=hi-lo,rh=rg/5,mg=Math.pow(10,Math.floor(Math.log10(rh)));const st=[1,2,5,10].find(s=>s*mg>=rh)*mg;const tk=[];let t=Math.ceil(lo/st)*st;while(t<=hi){tk.push(t);t+=st;}return{fn,min:lo,max:hi,ticks:tk};}
function CF({xs,ys,xL,yL,w=350,h=250,children}){const pw=w-PAD.left-PAD.right,ph=h-PAD.top-PAD.bottom,sxx=sc(xs,pw),syy=sc(ys,ph);return(<svg viewBox={`0 0 ${w} ${h}`} style={{width:"100%",height:"auto",display:"block"}}><g transform={`translate(${PAD.left},${PAD.top})`}>{sxx.ticks.map(t=><line key={`gx${t}`} x1={sxx.fn(t)} x2={sxx.fn(t)} y1={0} y2={ph} stroke={C.grid} strokeWidth={.6}/>)}{syy.ticks.map(t=><line key={`gy${t}`} x1={0} x2={pw} y1={ph-syy.fn(t)} y2={ph-syy.fn(t)} stroke={C.grid} strokeWidth={.6}/>)}<line x1={0} x2={pw} y1={ph} y2={ph} stroke={C.border}/><line x1={0} x2={0} y1={0} y2={ph} stroke={C.border}/>{sxx.ticks.map(t=><text key={`tx${t}`} x={sxx.fn(t)} y={ph+17} textAnchor="middle" fontSize={11} fill={C.sub}>{+t.toFixed(1)}</text>)}{syy.ticks.map(t=><text key={`ty${t}`} x={-8} y={ph-syy.fn(t)+4} textAnchor="end" fontSize={11} fill={C.sub}>{+t.toFixed(1)}</text>)}<text x={pw/2} y={ph+36} textAnchor="middle" fontSize={12} fill={C.text} fontWeight={600}>{xL}</text><text x={-38} y={ph/2} textAnchor="middle" fontSize={12} fill={C.text} fontWeight={600} transform={`rotate(-90,-38,${ph/2})`}>{yL}</text>{children(sxx.fn,vv=>ph-syy.fn(vv),pw,ph,sxx,syy)}</g></svg>);}

/* ── SECTION HEADERS + ICONS ─────────────────────────────────────────── */
function IconGlyph({type,size=19}){
  const common={width:size,height:size,viewBox:"0 0 24 24",fill:"none",stroke:"currentColor",strokeWidth:1.8,strokeLinecap:"round",strokeLinejoin:"round",style:{flex:"0 0 auto",opacity:.92}};
  if(type==="meaning")return <svg {...common}><circle cx="12" cy="12" r="8"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>;
  if(type==="assess")return <svg {...common}><path d="M4 19V5"/><path d="M4 19h16"/><path d="M7 15l3-4 3 2 4-6"/><circle cx="7" cy="15" r=".8"/><circle cx="10" cy="11" r=".8"/><circle cx="13" cy="13" r=".8"/><circle cx="17" cy="7" r=".8"/></svg>;
  if(type==="warning")return <svg {...common}><path d="M12 3l9 16H3L12 3z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>;
  if(type==="fix")return <svg {...common}><path d="M14.7 6.3a4 4 0 0 0-5 5L4 17v3h3l5.7-5.7a4 4 0 0 0 5-5"/><path d="M15 5l4 4"/></svg>;
  if(type==="rules")return <svg {...common}><path d="M5 4h14v16H5z"/><path d="M8 8h8"/><path d="M8 12h8"/><path d="M8 16h5"/></svg>;
  if(type==="types")return <svg {...common}><path d="M7 7h10"/><path d="M7 17h10"/><circle cx="7" cy="7" r="3"/><circle cx="17" cy="17" r="3"/></svg>;
  if(type==="example")return <svg {...common}><path d="M4 17l5-5 4 3 7-8"/><path d="M4 20h16"/></svg>;
  if(type==="reference")return <svg {...common}><path d="M6 4h9l3 3v13H6z"/><path d="M14 4v4h4"/><path d="M9 13h6"/><path d="M9 17h5"/></svg>;
  return <svg {...common}><circle cx="12" cy="12" r="8"/><path d="M8 12h8"/></svg>;
}
function iconTypeForHeading(children){
  const t=String(children).toLowerCase();
  if(t.includes("what it means")||t.includes("what is happening"))return "meaning";
  if(t.includes("assess")||t.includes("formal")||t.includes("reading"))return "assess";
  if(t.includes("wrong")||t.includes("violated"))return "warning";
  if(t.includes("fix"))return "fix";
  if(t.includes("rule"))return "rules";
  if(t.includes("type"))return "types";
  if(t.includes("example"))return "example";
  if(t.includes("reference"))return "reference";
  return "meaning";
}
function Hd({children}){return <div style={{display:"flex",alignItems:"center",gap:8,fontSize:15,fontWeight:700,color:"var(--accent, #3D3832)",marginBottom:7,borderLeft:"3px solid currentColor",paddingLeft:10}}>
  <IconGlyph type={iconTypeForHeading(children)}/><span>{children}</span>
</div>;}
function Lnk({href,children}){return <a href={href} target="_blank" rel="noopener noreferrer" style={{color:"#2B6CB0",textDecoration:"underline"}}>{children}</a>;}

/* ── MINI ICONS ──────────────────────────────────────────────────────── */
function MiniIcon({type}){const s={width:38,height:22,display:"block",margin:"4px auto 0"};
if(type==="linearity")return <svg viewBox="0 0 36 22" style={s}><line x1="2" x2="34" y1="11" y2="11" stroke="#2B6CB0" strokeWidth=".7" strokeDasharray="2,1.5"/>{[[5,8],[9,14],[13,9],[17,13],[21,7],[25,12],[29,10],[33,14]].map(([x,y],i)=><circle key={i} cx={x} cy={y} r="1.8" fill="#2B6CB0" opacity=".6"/>)}</svg>;
if(type==="homogeneity")return <svg viewBox="0 0 36 22" style={s}><path d="M3,14 Q18,8 33,6" fill="none" stroke="#2F855A" strokeWidth="1"/>{[[4,13],[8,12],[12,14],[16,10],[20,13],[24,8],[28,11],[32,5]].map(([x,y],i)=><circle key={i} cx={x} cy={y} r="1.8" fill="#2F855A" opacity=".6"/>)}</svg>;
if(type==="influential")return <svg viewBox="0 0 36 22" style={s}><line x1="2" x2="34" y1="11" y2="11" stroke="#999" strokeWidth=".4" strokeDasharray="1.5,1.5"/><path d="M3,4 Q15,9 33,10" fill="none" stroke="#C53030" strokeWidth=".7" strokeDasharray="2,1.5"/><path d="M3,18 Q15,13 33,12" fill="none" stroke="#C53030" strokeWidth=".7" strokeDasharray="2,1.5"/>{[[7,12],[10,9],[13,13],[16,11],[19,10],[22,12],[25,11]].map(([x,y],i)=><circle key={i} cx={x} cy={y} r="1.4" fill="#3B7DD8" opacity=".6"/>)}<circle cx="31" cy="3.5" r="2" fill="#C53030"/></svg>;
if(type==="normality")return <svg viewBox="0 0 36 22" style={s}><line x1="3" y1="11" x2="33" y2="11" stroke="#6B46C1" strokeWidth=".9"/>{[[5,14],[9,9],[13,12],[17,10],[21,11],[25,8],[29,13],[33,9]].map(([x,y],i)=><circle key={i} cx={x} cy={y} r="1.6" fill="#6B46C1" opacity=".65"/>)}</svg>;
return null;}

/* ── MCQ COMPONENT ───────────────────────────────────────────────────── */
function QuizVisual({type,color="#2B6CB0"}){
  const s={width:"100%",maxWidth:300,height:"auto",display:"block",margin:"0 auto 8px",background:C.panel,borderRadius:10,border:`1px solid ${C.border}`,boxShadow:"0 2px 8px rgba(0,0,0,.04)"};
  const SM=C.smooth,RF=C.ref,SB=C.sub,DH=C.dash,GR=C.grid,BD=C.border;

  /* ── SAMPLE SIZE ───────────────────────────────────────────── */
  // sample: same true line, single new red point shifts n=8 line a lot, n=80 line barely
  if(type==="sample")return <svg viewBox="0 0 300 120" style={s}>
    <text x="70" y="14" textAnchor="middle" fontSize="11" fill={SB} fontWeight="700">n = 8</text>
    <text x="226" y="14" textAnchor="middle" fontSize="11" fill={SB} fontWeight="700">n = 80</text>
    <rect x="14" y="20" width="116" height="76" fill="none" stroke={BD} rx="4"/>
    <rect x="170" y="20" width="116" height="76" fill="none" stroke={BD} rx="4"/>
    {[[28,72],[42,60],[56,76],[74,62],[88,52],[104,68],[118,58]].map(([x,y],i)=><circle key={`l${i}`} cx={x} cy={y} r="3.2" fill={color} opacity=".8"/>)}
    <line x1="18" y1="68" x2="126" y2="62" stroke={DH} strokeWidth="1.5" strokeDasharray="3,3" opacity=".7"/>
    <line x1="18" y1="78" x2="126" y2="36" stroke={SM} strokeWidth="2.2"/>
    <circle cx="120" cy="32" r="5" fill={RF} stroke="#fff" strokeWidth="1"/>
    {Array.from({length:32},(_,i)=>{const x=176+(i%8)*14+(Math.floor(i/8)%2)*4;const y=46+Math.floor(i/8)*10+(i*53%14-7);return [x,y];}).map(([x,y],i)=><circle key={`r${i}`} cx={x} cy={y} r="2.4" fill={color} opacity=".6"/>)}
    <line x1="174" y1="64" x2="282" y2="56" stroke={DH} strokeWidth="1.5" strokeDasharray="3,3" opacity=".7"/>
    <line x1="174" y1="65" x2="282" y2="54" stroke={SM} strokeWidth="2.2"/>
    <circle cx="276" cy="32" r="5" fill={RF} stroke="#fff" strokeWidth="1"/>
    <text x="150" y="111" textAnchor="middle" fontSize="9.5" fill={SB}>same red point added to each panel</text>
  </svg>;

  // sample_ci: same point estimate, two CI widths (sample size hidden)
  if(type==="sample_ci")return <svg viewBox="0 0 300 100" style={s}>
    <text x="150" y="14" textAnchor="middle" fontSize="11" fill={SB}>same slope estimate \u00B1 CI</text>
    <line x1="50" x2="280" y1="50" y2="50" stroke={GR} strokeDasharray="3,3"/>
    <line x1="62" x2="240" y1="38" y2="38" stroke={color} strokeWidth="2.4"/>
    <line x1="62" x2="62" y1="33" y2="43" stroke={color} strokeWidth="2.4"/>
    <line x1="240" x2="240" y1="33" y2="43" stroke={color} strokeWidth="2.4"/>
    <circle cx="151" cy="38" r="4" fill={color}/>
    <line x1="124" x2="178" y1="68" y2="68" stroke={color} strokeWidth="2.4"/>
    <line x1="124" x2="124" y1="63" y2="73" stroke={color} strokeWidth="2.4"/>
    <line x1="178" x2="178" y1="63" y2="73" stroke={color} strokeWidth="2.4"/>
    <circle cx="151" cy="68" r="4" fill={color}/>
  </svg>;

  // sample_stability: true line vs two small-n fits vs one large-n fit
  if(type==="sample_stability")return <svg viewBox="0 0 300 110" style={s}>
    <line x1="22" y1="86" x2="282" y2="86" stroke={BD}/>
    <line x1="22" y1="14" x2="22" y2="86" stroke={BD}/>
    <line x1="24" y1="76" x2="278" y2="22" stroke={DH} strokeWidth="2" strokeDasharray="5,4" opacity=".7"/>
    <line x1="24" y1="84" x2="278" y2="14" stroke={RF} strokeWidth="1.5" opacity=".75"/>
    <line x1="24" y1="62" x2="278" y2="38" stroke={RF} strokeWidth="1.5" opacity=".75"/>
    <line x1="24" y1="74" x2="278" y2="24" stroke={color} strokeWidth="2.2"/>
    <rect x="158" y="10" width="120" height="38" fill="#fff" stroke={BD} rx="3"/>
    <line x1="166" y1="20" x2="184" y2="20" stroke={DH} strokeWidth="2" strokeDasharray="3,3"/>
    <text x="190" y="23" fontSize="9.5" fill={SB}>true</text>
    <line x1="166" y1="32" x2="184" y2="32" stroke={RF} strokeWidth="2"/>
    <text x="190" y="35" fontSize="9.5" fill={SB}>two small-n fits</text>
    <line x1="166" y1="44" x2="184" y2="44" stroke={color} strokeWidth="2.2"/>
    <text x="190" y="47" fontSize="9.5" fill={SB}>one large-n fit</text>
    <text x="152" y="105" textAnchor="middle" fontSize="9.5" fill={SB}>different samples → different fitted lines</text>
  </svg>;

  // sample_residual_ok: a clean-looking residual plot from a tiny sample (n=8)
  if(type==="sample_residual_ok")return <svg viewBox="0 0 300 100" style={s}>
    <line x1="22" y1="80" x2="282" y2="80" stroke={BD}/>
    <line x1="22" y1="14" x2="22" y2="80" stroke={BD}/>
    <line x1="22" x2="282" y1="46" y2="46" stroke={DH} strokeDasharray="4,3" opacity=".55"/>
    {[[50,40],[82,52],[114,42],[146,50],[178,38],[210,52],[242,44],[270,48]].map(([x,y],i)=><circle key={i} cx={x} cy={y} r="3.5" fill={color} opacity=".8"/>)}
    <text x="152" y="20" textAnchor="middle" fontSize="11" fill={SB} fontWeight="700">residuals vs fitted, n = 8</text>
    <text x="14" y="48" textAnchor="middle" fontSize="10" fill={SB} transform="rotate(-90,14,48)">resid.</text>
    <text x="152" y="94" textAnchor="middle" fontSize="9.5" fill={SB}>fitted \u2192</text>
  </svg>;

  /* ── UNCORRELATED ──────────────────────────────────────────── */
  // time: residuals in time order with runs
  if(type==="time")return <svg viewBox="0 0 300 100" style={s}>
    <line x1="22" y1="84" x2="282" y2="84" stroke={BD}/>
    <line x1="22" y1="14" x2="22" y2="84" stroke={BD}/>
    <line x1="22" x2="282" y1="50" y2="50" stroke={DH} strokeDasharray="3,3" opacity=".5"/>
    {[1.0,1.2,0.9,1.1,0.4,-0.4,-0.9,-1.1,-0.7,-0.2,0.5,0.9,1.1,1.0].map((r,i)=>{const x=34+i*18,y=50-r*22;return <g key={i}><line x1={x} x2={x} y1={50} y2={y} stroke={color} strokeWidth="1.5" opacity=".4"/><circle cx={x} cy={y} r="3.2" fill={color}/></g>;})}
    <text x="152" y="98" textAnchor="middle" fontSize="9.5" fill={SB}>residuals plotted in collection order</text>
  </svg>;

  // cluster: three clinics with within-clinic correlation
  if(type==="cluster")return <svg viewBox="0 0 300 110" style={s}>
    {[60,150,240].map((cx,ci)=>{const off=[12,-14,4][ci];return <g key={ci}>
      <rect x={cx-36} y="14" width="72" height="64" rx="6" fill={color} opacity=".09" stroke={color} strokeOpacity=".35"/>
      {[0,1,2,3,4,5].map(j=><circle key={j} cx={cx-25+j*10} cy={46+off+[3,-2,2,-3,1,-1][j]} r="3" fill={color} opacity=".8"/>)}
      <text x={cx} y="98" textAnchor="middle" fontSize="11" fill={SB} fontWeight="700">clinic {ci+1}</text>
    </g>;})}
  </svg>;

  // hidden_time: clean vs fitted, but wavy when ordered by time
  if(type==="hidden_time")return <svg viewBox="0 0 300 110" style={s}>
    <text x="76" y="14" textAnchor="middle" fontSize="11" fill={SB} fontWeight="700">vs fitted</text>
    <text x="226" y="14" textAnchor="middle" fontSize="11" fill={SB} fontWeight="700">vs collection time</text>
    <rect x="14" y="20" width="124" height="68" fill="none" stroke={BD} rx="4"/>
    <rect x="162" y="20" width="124" height="68" fill="none" stroke={BD} rx="4"/>
    <line x1="18" y1="54" x2="134" y2="54" stroke={DH} strokeDasharray="3,3" opacity=".5"/>
    <line x1="166" y1="54" x2="282" y2="54" stroke={DH} strokeDasharray="3,3" opacity=".5"/>
    {[[26,46],[40,62],[54,48],[68,60],[84,42],[98,56],[112,46],[128,60]].map(([x,y],i)=><circle key={`f${i}`} cx={x} cy={y} r="3" fill={color} opacity=".8"/>)}
    {Array.from({length:11},(_,i)=>[168+i*11,54-Math.sin(i*0.7)*16]).map(([x,y],i)=><circle key={`t${i}`} cx={x} cy={y} r="3" fill={color} opacity=".85"/>)}
    <text x="150" y="103" textAnchor="middle" fontSize="9.5" fill={SB}>same residuals, two views</text>
  </svg>;

  // four_diag: the four standard residual plots, all clean
  if(type==="four_diag")return <svg viewBox="0 0 300 110" style={s}>
    {[[8,14],[156,14],[8,62],[156,62]].map(([x,y],i)=><g key={i}>
      <rect x={x} y={y} width="136" height="42" fill="none" stroke={BD} rx="3"/>
      <line x1={x+4} y1={y+22} x2={x+132} y2={y+22} stroke={DH} strokeDasharray="2,2" opacity=".4"/>
      {[10,24,40,58,76,94,112,126].map((dx,j)=><circle key={j} cx={x+dx} cy={y+22+([2,-3,1,-2,3,-2,2,-1][j])} r="1.7" fill={color} opacity=".75"/>)}
      <text x={x+6} y={y+10} fontSize="8.5" fill={SB} fontWeight="700">{["Linearity","Homogeneity","Influence","Q-Q"][i]}</text>
    </g>)}
    <text x="150" y="105" textAnchor="middle" fontSize="9.5" fill={SB}>not every assumption shows up here</text>
  </svg>;

  /* ── MULTICOLLINEARITY ─────────────────────────────────────── */
  // one_predictor: just one X, no possibility of collinearity
  if(type==="one_predictor")return <svg viewBox="0 0 300 100" style={s}>
    <text x="22" y="20" fontSize="11" fill={SB} fontWeight="700">predictors in this model:</text>
    <circle cx="90" cy="58" r="32" fill={color} opacity=".18" stroke={color} strokeWidth="2"/>
    <text x="90" y="64" textAnchor="middle" fontSize="18" fill={color} fontWeight="800">X</text>
    <circle cx="218" cy="58" r="32" fill="#fff" stroke={BD} strokeDasharray="4,4" strokeWidth="1.5"/>
    <text x="218" y="62" textAnchor="middle" fontSize="22" fill={BD} fontWeight="800">?</text>
  </svg>;

  // vif: overlapping ellipses showing predictors share information
  if(type==="vif")return <svg viewBox="0 0 300 100" style={s}>
    <ellipse cx="115" cy="50" rx="72" ry="32" fill={color} opacity=".22" stroke={color} strokeWidth="1.5"/>
    <ellipse cx="195" cy="50" rx="72" ry="32" fill={RF} opacity=".18" stroke={RF} strokeWidth="1.5"/>
    <text x="70" y="55" textAnchor="middle" fontSize="11" fill={color} fontWeight="700">education</text>
    <text x="240" y="55" textAnchor="middle" fontSize="11" fill={RF} fontWeight="700">income</text>
    <text x="155" y="92" textAnchor="middle" fontSize="9.5" fill={SB}>two predictors in the model</text>
  </svg>;

  // coef_uncertain: two coefficients with very wide CIs crossing zero
  if(type==="coef_uncertain")return <svg viewBox="0 0 300 100" style={s}>
    <text x="150" y="14" textAnchor="middle" fontSize="11" fill={SB}>coefficient ± CI</text>
    <line x1="150" x2="150" y1="22" y2="80" stroke={SB} strokeDasharray="3,3" opacity=".4"/>
    <text x="156" y="22" fontSize="9" fill={SB}>0</text>
    <line x1="60" x2="232" y1="40" y2="40" stroke={color} strokeWidth="2.4"/>
    <line x1="60" x2="60" y1="35" y2="45" stroke={color} strokeWidth="2.4"/>
    <line x1="232" x2="232" y1="35" y2="45" stroke={color} strokeWidth="2.4"/>
    <circle cx="120" cy="40" r="4" fill={color}/>
    <text x="22" y="44" fontSize="11" fill={SB} fontWeight="700">β₁</text>
    <line x1="80" x2="252" y1="68" y2="68" stroke={RF} strokeWidth="2.4"/>
    <line x1="80" x2="80" y1="63" y2="73" stroke={RF} strokeWidth="2.4"/>
    <line x1="252" x2="252" y1="63" y2="73" stroke={RF} strokeWidth="2.4"/>
    <circle cx="172" cy="68" r="4" fill={RF}/>
    <text x="22" y="72" fontSize="11" fill={SB} fontWeight="700">β₂</text>
    <text x="150" y="94" textAnchor="middle" fontSize="9.5" fill={SB}>wide intervals: both could include zero</text>
  </svg>;

  // collinear_scatter: two predictors that move together
  if(type==="collinear_scatter")return <svg viewBox="0 0 300 100" style={s}>
    <line x1="22" y1="84" x2="282" y2="84" stroke={BD}/>
    <line x1="22" y1="14" x2="22" y2="84" stroke={BD}/>
    {Array.from({length:14},(_,i)=>{const x=40+i*16+(i%2?2:-2);const y=78-i*4.5+(i*53%6-3);return <circle key={i} cx={x} cy={y} r="3.2" fill={color} opacity=".8"/>;})}
    <line x1="30" y1="80" x2="278" y2="20" stroke={SM} strokeWidth="2" opacity=".7"/>
    <text x="152" y="98" textAnchor="middle" fontSize="9.5" fill={SB}>education vs income: strongly correlated</text>
  </svg>;

  /* ── SHAPE / LINEARITY ─────────────────────────────────────── */
  // curve: monotone concave scatter with a straight OLS line that misses the curvature
  if(type==="curve")return <svg viewBox="0 0 300 100" style={s}>
    <line x1="22" y1="84" x2="282" y2="84" stroke={BD}/>
    <line x1="22" y1="14" x2="22" y2="84" stroke={BD}/>
    {Array.from({length:13},(_,i)=>{const t=i/12;const x=32+i*20;const y=80-54*Math.sqrt(t)+(i%2?2:-2);return <circle key={i} cx={x} cy={y} r="3.2" fill={color} opacity=".78"/>;})}
    <line x1="22" y1="70" x2="282" y2="20" stroke={RF} strokeWidth="2.2"/>
    <text x="152" y="98" textAnchor="middle" fontSize="9.5" fill={SB}>straight line cannot follow the curve</text>
  </svg>;

  // dose_response: rising curve with steep low-dose tangent and flat high-dose tangent
  if(type==="dose_response")return <svg viewBox="0 0 300 100" style={s}>
    <line x1="28" y1="80" x2="282" y2="80" stroke={BD}/>
    <line x1="28" y1="14" x2="28" y2="80" stroke={BD}/>
    <path d="M 28 76 Q 90 22 282 18" fill="none" stroke={color} strokeWidth="2.4"/>
    <line x1="34" y1="74" x2="92" y2="36" stroke={SM} strokeWidth="1.6" strokeDasharray="3,3"/>
    <line x1="200" y1="22" x2="278" y2="18" stroke={SM} strokeWidth="1.6" strokeDasharray="3,3"/>
    <text x="62" y="96" textAnchor="middle" fontSize="9.5" fill={SB}>low dose</text>
    <text x="240" y="96" textAnchor="middle" fontSize="9.5" fill={SB}>high dose</text>
    <text x="14" y="48" textAnchor="middle" fontSize="10" fill={SB} transform="rotate(-90,14,48)">relief</text>
  </svg>;

  // resid_curve: residuals make a U-shape (or arch)
  if(type==="resid_curve")return <svg viewBox="0 0 300 100" style={s}>
    <line x1="22" y1="84" x2="282" y2="84" stroke={BD}/>
    <line x1="22" y1="14" x2="22" y2="84" stroke={BD}/>
    <line x1="22" x2="282" y1="50" y2="50" stroke={DH} strokeDasharray="4,3" opacity=".55"/>
    <path d="M 28 24 Q 152 86 276 24" fill="none" stroke={SM} strokeWidth="2.5"/>
    {Array.from({length:13},(_,i)=>{const x=34+i*20;const u=Math.cos((i/12)*Math.PI*2)*0.5+0.5;const y=24+(64-24)*(1-u)+(i*17%6-3);return <circle key={i} cx={x} cy={y} r="3" fill={color} opacity=".75"/>;})}
    <text x="14" y="48" textAnchor="middle" fontSize="10" fill={SB} transform="rotate(-90,14,48)">resid.</text>
    <text x="152" y="98" textAnchor="middle" fontSize="9.5" fill={SB}>fitted →</text>
  </svg>;

  // resid_good: random scatter, flat smoother
  if(type==="resid_good")return <svg viewBox="0 0 300 100" style={s}>
    <line x1="22" y1="84" x2="282" y2="84" stroke={BD}/>
    <line x1="22" y1="14" x2="22" y2="84" stroke={BD}/>
    <line x1="22" x2="282" y1="50" y2="50" stroke={DH} strokeDasharray="4,3" opacity=".55"/>
    <path d="M 22 50 Q 152 51 282 49" fill="none" stroke={SM} strokeWidth="2.5"/>
    {Array.from({length:13},(_,i)=>{const x=34+i*20;const y=50+([4,-5,2,-3,5,-2,3,-4,5,-3,2,-4,3][i]);return <circle key={i} cx={x} cy={y} r="3" fill={color} opacity=".75"/>;})}
    <text x="14" y="48" textAnchor="middle" fontSize="10" fill={SB} transform="rotate(-90,14,48)">resid.</text>
    <text x="152" y="98" textAnchor="middle" fontSize="9.5" fill={SB}>fitted →</text>
  </svg>;

  // log_fix: before (curve) and after log transform (straight)
  if(type==="log_fix")return <svg viewBox="0 0 300 110" style={s}>
    <text x="76" y="14" textAnchor="middle" fontSize="11" fill={SB} fontWeight="700">y vs x</text>
    <text x="226" y="14" textAnchor="middle" fontSize="11" fill={SB} fontWeight="700">log(y) vs x</text>
    <rect x="14" y="20" width="124" height="68" fill="none" stroke={BD} rx="4"/>
    <rect x="162" y="20" width="124" height="68" fill="none" stroke={BD} rx="4"/>
    <path d="M 18 84 Q 80 22 134 24" fill="none" stroke={color} strokeWidth="2"/>
    {[[28,76],[42,58],[58,42],[78,32],[100,28],[124,26]].map(([x,y],i)=><circle key={`a${i}`} cx={x} cy={y} r="3" fill={color}/>)}
    <line x1="166" y1="80" x2="282" y2="28" stroke={color} strokeWidth="2"/>
    {[[174,76],[192,66],[208,56],[226,46],[244,38],[262,30]].map(([x,y],i)=><circle key={`b${i}`} cx={x} cy={y} r="3" fill={color}/>)}
    <text x="76" y="103" textAnchor="middle" fontSize="9.5" fill={SB}>curved</text>
    <text x="226" y="103" textAnchor="middle" fontSize="9.5" fill={SM} fontWeight="700">straightened</text>
  </svg>;

  /* ── HOMOGENEITY ───────────────────────────────────────────── */
  // fan: classic fan/cone in scatter
  if(type==="fan")return <svg viewBox="0 0 300 100" style={s}>
    <line x1="22" y1="84" x2="282" y2="84" stroke={BD}/>
    <line x1="22" y1="14" x2="22" y2="84" stroke={BD}/>
    <line x1="22" y1="62" x2="282" y2="34" stroke={SM} strokeWidth="2"/>
    {Array.from({length:13},(_,i)=>{const x=32+i*20;const center=62-(i*28/12);const spread=2+i*2.4;return <g key={i}><circle cx={x} cy={center+spread} r="2.8" fill={color} opacity=".65"/><circle cx={x} cy={center-spread} r="2.8" fill={color} opacity=".65"/></g>;})}
    <text x="152" y="98" textAnchor="middle" fontSize="9.5" fill={SB}>x →</text>
  </svg>;

  // scale_loc: LOESS-like curvy smoother with shaded CI band, mirrors main DiagPlot homogeneity panel
  if(type==="scale_loc")return <svg viewBox="0 0 300 100" style={s}>
    <line x1="22" y1="80" x2="282" y2="80" stroke={BD}/>
    <line x1="22" y1="14" x2="22" y2="80" stroke={BD}/>
    <path d="M 22 76 Q 80 70 130 60 Q 200 42 282 22 L 282 30 Q 200 50 130 68 Q 80 78 22 84 Z" fill={SM} opacity=".14"/>
    {Array.from({length:14},(_,i)=>{const x=32+i*19;const yC=72-Math.pow(i/13,1.4)*50;const noise=[5,-4,2,-3,4,-2,3,5,-3,4,-2,2,-3,4][i];return <circle key={i} cx={x} cy={yC+noise} r="3" fill={color} opacity=".7"/>;})}
    <path d="M 22 72 Q 80 66 130 56 Q 200 36 282 18" fill="none" stroke={SM} strokeWidth="2.5"/>
    <text x="14" y="48" textAnchor="middle" fontSize="10" fill={SB} transform="rotate(-90,14,48)">√|std r|</text>
    <text x="152" y="96" textAnchor="middle" fontSize="9.5" fill={SB}>fitted →</text>
  </svg>;

  // het_pick: three small panels: homoscedastic, fan (heteroscedastic), curved (linearity issue)
  if(type==="het_pick")return <svg viewBox="0 0 300 132" style={s}>
    {[0,1,2].map(p=>{const x0=8+p*98;return <g key={p}>
      <text x={x0+44} y={14} textAnchor="middle" fontSize="10" fill={SB} fontWeight="700">{["left","middle","right"][p]}</text>
      <rect x={x0} y={20} width={88} height={70} fill="none" stroke={BD} rx={3}/>
      <line x1={x0+8} x2={x0+82} y1={55} y2={55} stroke={DH} strokeDasharray="3,2" opacity=".5"/>
      <text x={x0+44} y={104} textAnchor="middle" fontSize="9" fill={SB}>fitted</text>
      <text x={x0-1} y={56} fontSize="8.5" fill={SB} transform={`rotate(-90,${x0-1},56)`} textAnchor="middle">resid.</text>
    </g>;})}
    {[0,12,24,36,48,60,72].map((dx,i)=>{const x=14+dx;const y=55+([4,-5,3,-3,5,-2,4][i])*2.4;return <circle key={`a${i}`} cx={x} cy={y} r="2.5" fill={color} opacity=".78"/>;})}
    {[0,12,24,36,48,60,72].map((dx,i)=>{const x=112+dx;const sp=2+i*2.5;const j=([1,-1,1,-1,1,-1,1])[i];return <g key={`b${i}`}><circle cx={x} cy={55+sp*j} r="2.5" fill={color} opacity=".75"/><circle cx={x} cy={55-sp*j*0.7} r="2.5" fill={color} opacity=".75"/></g>;})}
    {[0,12,24,36,48,60,72].map((dx,i)=>{const x=210+dx;const u=Math.cos((i/6)*Math.PI)*16;const y=55-u+([2,-2,1,-1,2,-1,1][i]);return <circle key={`c${i}`} cx={x} cy={y} r="2.5" fill={color} opacity=".8"/>;})}
  </svg>;

  // het_se_problem: naive (too narrow) vs robust (honest, wider) CI
  if(type==="het_se_problem")return <svg viewBox="0 0 300 100" style={s}>
    <text x="150" y="14" textAnchor="middle" fontSize="11" fill={SB}>same coefficient, different CIs</text>
    <line x1="50" x2="280" y1="50" y2="50" stroke={GR} strokeDasharray="3,3"/>
    <line x1="124" x2="178" y1="38" y2="38" stroke={color} strokeWidth="2.4"/>
    <line x1="124" x2="124" y1="33" y2="43" stroke={color} strokeWidth="2.4"/>
    <line x1="178" x2="178" y1="33" y2="43" stroke={color} strokeWidth="2.4"/>
    <circle cx="151" cy="38" r="4" fill={color}/>
    <text x="22" y="42" fontSize="11" fill={SB}>naive</text>
    <text x="184" y="42" fontSize="9.5" fill={RF} fontWeight="700">too narrow</text>
    <line x1="80" x2="222" y1="68" y2="68" stroke={SM} strokeWidth="2.6"/>
    <line x1="80" x2="80" y1="63" y2="73" stroke={SM} strokeWidth="2.6"/>
    <line x1="222" x2="222" y1="63" y2="73" stroke={SM} strokeWidth="2.6"/>
    <circle cx="151" cy="68" r="4" fill={SM}/>
    <text x="22" y="72" fontSize="11" fill={SM} fontWeight="700">robust</text>
    <text x="228" y="72" fontSize="9.5" fill={SB}>honest width</text>
  </svg>;

  // fan_vs_skew: scatter fan (heteroscedastic) vs skewed density (non-normal)
  if(type==="fan_vs_skew")return <svg viewBox="0 0 300 110" style={s}>
    <text x="76" y="14" textAnchor="middle" fontSize="11" fill={SB} fontWeight="700">heteroscedasticity</text>
    <text x="226" y="14" textAnchor="middle" fontSize="11" fill={SB} fontWeight="700">non-normality</text>
    <rect x="14" y="20" width="124" height="68" fill="none" stroke={BD} rx="4"/>
    <rect x="162" y="20" width="124" height="68" fill="none" stroke={BD} rx="4"/>
    <line x1="18" y1="60" x2="134" y2="48" stroke={SM} strokeWidth="1.6" opacity=".7"/>
    {Array.from({length:9},(_,i)=>{const x=24+i*13,c=60-i*1.4,sp=2+i*2.6;return <g key={i}><circle cx={x} cy={c+sp} r="2.4" fill={color} opacity=".65"/><circle cx={x} cy={c-sp} r="2.4" fill={color} opacity=".65"/></g>;})}
    <line x1="166" y1="86" x2="282" y2="86" stroke={BD}/>
    <path d="M 168 86 C 184 86, 200 28, 220 28 C 245 28, 256 64, 282 84 L 282 86 Z" fill={color} opacity=".25" stroke={color}/>
    <text x="76" y="105" textAnchor="middle" fontSize="9.5" fill={SB}>spread changes with X</text>
    <text x="226" y="105" textAnchor="middle" fontSize="9.5" fill={SB}>residual shape is skewed</text>
  </svg>;

  // income_fan: scatter where y axis is dollars, fan growing with income
  if(type==="income_fan")return <svg viewBox="0 0 300 100" style={s}>
    <line x1="32" y1="80" x2="282" y2="80" stroke={BD}/>
    <line x1="32" y1="14" x2="32" y2="80" stroke={BD}/>
    {Array.from({length:14},(_,i)=>{const x=42+i*18;const center=70-(i*32/13);const sp=1.5+i*2.4;return <g key={i}><circle cx={x} cy={center+sp} r="2.6" fill={color} opacity=".65"/><circle cx={x} cy={center-sp} r="2.6" fill={color} opacity=".65"/></g>;})}
    <line x1="32" y1="70" x2="282" y2="38" stroke={SM} strokeWidth="2"/>
    <text x="14" y="48" textAnchor="middle" fontSize="10" fill={SB} transform="rotate(-90,14,48)">$ spent</text>
    <text x="155" y="96" textAnchor="middle" fontSize="9.5" fill={SB}>income →  spending variability balloons</text>
  </svg>;

  /* ── INFLUENTIAL POINTS ────────────────────────────────────── */
  // outlier_types: 3 panels: outlier-low-lev, high-lev-on-trend, off-trend-high-lev
  if(type==="outlier_types")return <svg viewBox="0 0 300 110" style={s}>
    <rect x="6" y="22" width="92" height="64" fill="none" stroke={BD} rx="3"/>
    <line x1="10" y1="76" x2="94" y2="32" stroke={SM} strokeWidth="1.6"/>
    {[[20,68],[34,58],[48,48],[62,40],[80,30]].map(([x,y],i)=><circle key={i} cx={x} cy={y} r="2.6" fill={color}/>)}
    <circle cx="42" cy="60" r="4" fill={RF}/>
    <text x="52" y="16" textAnchor="middle" fontSize="9.5" fill={SB} fontWeight="700">left panel</text>
    <rect x="104" y="22" width="92" height="64" fill="none" stroke={BD} rx="3"/>
    <line x1="108" y1="76" x2="192" y2="32" stroke={SM} strokeWidth="1.6"/>
    {[[114,72],[126,64],[140,54],[152,46]].map(([x,y],i)=><circle key={i} cx={x} cy={y} r="2.6" fill={color}/>)}
    <circle cx="186" cy="34" r="4" fill={RF}/>
    <text x="150" y="16" textAnchor="middle" fontSize="9.5" fill={SB} fontWeight="700">middle panel</text>
    <rect x="202" y="22" width="92" height="64" fill="none" stroke={BD} rx="3"/>
    <line x1="206" y1="64" x2="290" y2="34" stroke={SM} strokeWidth="1.6"/>
    {[[214,68],[226,60],[240,52],[252,46]].map(([x,y],i)=><circle key={i} cx={x} cy={y} r="2.6" fill={color}/>)}
    <circle cx="284" cy="74" r="5" fill={RF}/>
    <text x="248" y="16" textAnchor="middle" fontSize="9.5" fill={SB} fontWeight="700">right panel</text>
  </svg>;

  // lev_resid: leverage vs std residual with Cook's contour, one influential point
  if(type==="lev_resid")return <svg viewBox="0 0 300 110" style={s}>
    <line x1="40" y1="86" x2="284" y2="86" stroke={BD}/>
    <line x1="40" y1="14" x2="40" y2="86" stroke={BD}/>
    <line x1="40" x2="284" y1="50" y2="50" stroke={DH} strokeDasharray="3,3" opacity=".5"/>
    <path d="M 40 18 Q 150 40 284 47" fill="none" stroke={SM} strokeDasharray="6,5" strokeWidth="1.4" opacity=".75"/>
    <path d="M 40 82 Q 150 60 284 53" fill="none" stroke={SM} strokeDasharray="6,5" strokeWidth="1.4" opacity=".75"/>
    {[[64,46],[78,56],[96,42],[112,54],[130,48],[148,52],[168,46],[188,48]].map(([x,y],i)=><circle key={i} cx={x} cy={y} r="2.8" fill={color} opacity=".78"/>)}
    <circle cx="262" cy="22" r="5" fill={RF}/>
    <text x="160" y="104" textAnchor="middle" fontSize="9.5" fill={SB}>leverage →</text>
    <text x="22" y="50" textAnchor="middle" fontSize="9.5" fill={SB} transform="rotate(-90,22,50)">std r</text>
    <text x="278" y="60" textAnchor="end" fontSize="9" fill={SM}>Cook's D = 0.5</text>
  </svg>;

  // cook_keep: Cook's distance bars with one tall flagged bar and a 4/n threshold
  if(type==="cook_keep")return <svg viewBox="0 0 300 100" style={s}>
    <line x1="28" y1="80" x2="282" y2="80" stroke={BD}/>
    <line x1="28" y1="14" x2="28" y2="80" stroke={BD}/>
    <line x1="28" x2="282" y1="40" y2="40" stroke={RF} strokeDasharray="4,3" opacity=".7"/>
    <text x="282" y="36" textAnchor="end" fontSize="9.5" fill={RF}>Cook's D threshold</text>
    {[8,12,7,14,9,11,8,15,10,13,9,11,46].map((h,i)=>{const x=42+i*18;const flagged=h>30;return <line key={i} x1={x} x2={x} y1={80} y2={80-h} stroke={flagged?RF:color} strokeWidth="6" strokeLinecap="round"/>;})}
    <text x="155" y="96" textAnchor="middle" fontSize="9.5" fill={SB}>observation index</text>
    <text x="14" y="48" textAnchor="middle" fontSize="10" fill={SB} transform="rotate(-90,14,48)">Cook's D</text>
  </svg>;

  // cooks_combine: leverage × residual = Cook's D
  if(type==="cooks_combine")return <svg viewBox="0 0 300 100" style={s}>
    <circle cx="62" cy="50" r="32" fill={color} opacity=".18" stroke={color}/>
    <text x="62" y="46" textAnchor="middle" fontSize="11" fill={color} fontWeight="700">leverage</text>
    <text x="62" y="60" textAnchor="middle" fontSize="9" fill={SB}>(unusual X)</text>
    <text x="120" y="56" textAnchor="middle" fontSize="22" fill={SB} fontWeight="600">×</text>
    <circle cx="178" cy="50" r="32" fill={color} opacity=".18" stroke={color}/>
    <text x="178" y="46" textAnchor="middle" fontSize="11" fill={color} fontWeight="700">residual²</text>
    <text x="178" y="60" textAnchor="middle" fontSize="9" fill={SB}>(off the line)</text>
    <path d="M 216 50 L 240 50" stroke={SM} strokeWidth="2.4" markerEnd="url(#arrCK)"/>
    <defs><marker id="arrCK" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill={SM}/></marker></defs>
    <rect x="244" y="32" width="48" height="36" rx="5" fill={SM} opacity=".18" stroke={SM}/>
    <text x="268" y="48" textAnchor="middle" fontSize="11" fill={SM} fontWeight="800">Cook's</text>
    <text x="268" y="62" textAnchor="middle" fontSize="11" fill={SM} fontWeight="800">D</text>
  </svg>;

  // slope_shift: same scatter, line with vs without an influential point
  if(type==="slope_shift")return <svg viewBox="0 0 300 100" style={s}>
    <line x1="22" y1="84" x2="282" y2="84" stroke={BD}/>
    <line x1="22" y1="14" x2="22" y2="84" stroke={BD}/>
    {[[42,72],[62,68],[84,65],[106,61],[128,58],[150,54],[172,51],[194,47],[216,44],[238,40]].map(([x,y],i)=><circle key={i} cx={x} cy={y} r="3.2" fill={color} opacity=".82"/>)}
    <circle cx="266" cy="18" r="5.2" fill={RF} stroke="#fff" strokeWidth="1"/>
    <line x1="36" y1="73" x2="248" y2="39" stroke={DH} strokeWidth="2.2" strokeDasharray="5,4" opacity=".82"/>
    <line x1="36" y1="77" x2="274" y2="20" stroke={RF} strokeWidth="2.4"/>
    <rect x="164" y="10" width="112" height="34" rx="4" fill="#fff" stroke={BD}/>
    <line x1="172" y1="21" x2="194" y2="21" stroke={DH} strokeWidth="2" strokeDasharray="4,3"/>
    <text x="200" y="24" fontSize="9.5" fill={SB}>without red</text>
    <line x1="172" y1="34" x2="194" y2="34" stroke={RF} strokeWidth="2.2"/>
    <text x="200" y="37" fontSize="9.5" fill={RF} fontWeight="700">with red</text>
    <text x="152" y="98" textAnchor="middle" fontSize="9.5" fill={SB}>one point reshapes the slope</text>
  </svg>;

  /* ── NORMALITY ─────────────────────────────────────────────── */
  // qq_heavy_tails: detrended Q-Q showing S-shape, heavy tails at both ends
  if(type==="qq_heavy_tails")return <svg viewBox="0 0 300 100" style={s}>
    <line x1="22" y1="84" x2="282" y2="84" stroke={BD}/>
    <line x1="22" y1="14" x2="22" y2="84" stroke={BD}/>
    <path d="M 30 38 Q 152 46 274 38 L 274 62 Q 152 54 30 62 Z" fill={SM} opacity=".13"/>
    <line x1="22" x2="282" y1="50" y2="50" stroke={SM} strokeWidth="2.2" opacity=".85"/>
    {[34,48,62,76,90,104,118,132,146,160,174,188,202,216,230,244,258,274].map((x,i)=>{
      const middleNoise=[1,3,-2,1,-3,2,-1,2,-2,3,-1,2,1,-2][i-2]||0;
      let dev;
      if(i===0)dev=-22;else if(i===1)dev=-14;else if(i===2)dev=-6;
      else if(i===17)dev=22;else if(i===16)dev=14;else if(i===15)dev=6;
      else dev=middleNoise;
      return <circle key={i} cx={x} cy={50+dev} r="2.7" fill={color} opacity=".78"/>;
    })}
    <text x="155" y="98" textAnchor="middle" fontSize="9.5" fill={SB}>detrended Q-Q · normal quantile →</text>
  </svg>;

  // worm: detrended Q-Q with random scatter inside a confidence band
  if(type==="worm")return <svg viewBox="0 0 300 100" style={s}>
    <line x1="22" y1="84" x2="282" y2="84" stroke={BD}/>
    <line x1="22" y1="14" x2="22" y2="84" stroke={BD}/>
    <path d="M 30 36 Q 152 44 274 36 L 274 64 Q 152 56 30 64 Z" fill={SM} opacity=".13"/>
    <line x1="22" x2="282" y1="50" y2="50" stroke={SM} strokeWidth="2.2" opacity=".85"/>
    {[34,48,62,76,90,104,118,132,146,160,174,188,202,216,230,244,258,274].map((x,i)=>{
      const dev=[5,-4,3,-2,4,-3,1,-4,2,-1,3,-3,4,-2,1,-3,2,-4][i];
      return <circle key={i} cx={x} cy={50+dev} r="2.7" fill={color} opacity=".78"/>;
    })}
    <text x="155" y="98" textAnchor="middle" fontSize="9.5" fill={SB}>detrended Q-Q · normal quantile →</text>
  </svg>;

  // tails: detrended Q-Q with red dots at the ends, blue middle
  if(type==="tails")return <svg viewBox="0 0 300 100" style={s}>
    <line x1="22" y1="84" x2="282" y2="84" stroke={BD}/>
    <line x1="22" y1="14" x2="22" y2="84" stroke={BD}/>
    <line x1="22" x2="282" y1="50" y2="50" stroke={SM} strokeWidth="2.2" opacity=".85"/>
    {[34,56,80,104,128,154,180,206,230,254,278].map((x,i)=>{const offTail=i<2||i>8;const dev=offTail?(i<2?16:20):[2,-3,1,-2,3,-1,2][i-2];return <circle key={i} cx={x} cy={50+dev} r="3" fill={offTail?RF:color} opacity={offTail ? .85 : .7}/>;})}
    <text x="155" y="98" textAnchor="middle" fontSize="9.5" fill={SB}>detrended Q-Q · normal quantile →</text>
  </svg>;

  // clt_safety: small-n irregular density vs large-n smooth bell
  if(type==="clt_safety")return <svg viewBox="0 0 300 110" style={s}>
    <text x="76" y="14" textAnchor="middle" fontSize="11" fill={SB} fontWeight="700">n = 10</text>
    <text x="226" y="14" textAnchor="middle" fontSize="11" fill={SB} fontWeight="700">n = 200</text>
    <line x1="14" y1="86" x2="138" y2="86" stroke={BD}/>
    <line x1="162" y1="86" x2="286" y2="86" stroke={BD}/>
    <path d="M 16 86 C 36 86, 44 32, 70 38 C 94 44, 100 72, 116 82 C 124 86, 134 84, 136 86 Z" fill={RF} opacity=".22" stroke={RF}/>
    <path d="M 164 86 C 182 86, 200 24, 224 24 C 252 24, 268 86, 284 86 Z" fill={color} opacity=".25" stroke={color}/>
    <text x="76" y="103" textAnchor="middle" fontSize="9.5" fill={RF}>residual shape really matters</text>
    <text x="226" y="103" textAnchor="middle" fontSize="9.5" fill={SM} fontWeight="700">CLT lends a safety net</text>
  </svg>;

  // pred_vs_inf: arrows showing prediction OK, CI/p-value affected
  if(type==="pred_vs_inf")return <svg viewBox="0 0 300 100" style={s}>
    <rect x="14" y="32" width="80" height="40" rx="6" fill={color} opacity=".15" stroke={color}/>
    <text x="54" y="48" textAnchor="middle" fontSize="11" fill={color} fontWeight="700">predictions</text>
    <text x="54" y="62" textAnchor="middle" fontSize="9.5" fill={SM} fontWeight="700">still OK</text>
    <line x1="100" y1="52" x2="124" y2="52" stroke={SM} strokeWidth="2" markerEnd="url(#arrPV)"/>
    <text x="156" y="48" textAnchor="middle" fontSize="11" fill={SB}>but</text>
    <line x1="180" y1="52" x2="204" y2="52" stroke={RF} strokeWidth="2" markerEnd="url(#arrPVx)"/>
    <rect x="208" y="32" width="80" height="40" rx="6" fill={RF} opacity=".15" stroke={RF}/>
    <text x="248" y="48" textAnchor="middle" fontSize="11" fill={RF} fontWeight="700">p-values, CIs</text>
    <text x="248" y="62" textAnchor="middle" fontSize="9.5" fill={RF} fontWeight="700">unreliable</text>
    <defs>
      <marker id="arrPV" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill={SM}/></marker>
      <marker id="arrPVx" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill={RF}/></marker>
    </defs>
  </svg>;

  /* ── EXOGENEITY ────────────────────────────────────────────── */
  // confound: injury severity → physical therapy visits + pain score
  if(type==="confound")return <svg viewBox="0 0 300 120" style={s}>
    <defs><marker id="arrCf" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill={color}/></marker></defs>
    <circle cx="150" cy="28" r="24" fill="#F0E4C8" stroke={color} strokeWidth="2"/>
    <text x="150" y="25" textAnchor="middle" fontSize="10" fill={color} fontWeight="700">injury</text>
    <text x="150" y="38" textAnchor="middle" fontSize="10" fill={color} fontWeight="700">severity</text>
    <circle cx="62" cy="86" r="24" fill="#fff" stroke={BD} strokeWidth="1.5"/>
    <text x="62" y="83" textAnchor="middle" fontSize="10" fill={SB}>therapy</text>
    <text x="62" y="96" textAnchor="middle" fontSize="10" fill={SB}>visits</text>
    <circle cx="238" cy="86" r="24" fill="#fff" stroke={BD} strokeWidth="1.5"/>
    <text x="238" y="90" textAnchor="middle" fontSize="10" fill={SB}>pain score</text>
    <path d="M 132 46 L 80 70" stroke={color} strokeWidth="2.2" markerEnd="url(#arrCf)"/>
    <path d="M 168 46 L 220 70" stroke={color} strokeWidth="2.2" markerEnd="url(#arrCf)"/>
    <line x1="86" y1="86" x2="214" y2="86" stroke={RF} strokeWidth="2" strokeDasharray="5,5"/>
    <text x="150" y="113" textAnchor="middle" fontSize="9.5" fill={SB}>observed correlation: visits ↔ pain</text>
  </svg>;

  // age_cars: wealth → age + cars
  if(type==="age_cars")return <svg viewBox="0 0 300 120" style={s}>
    <defs><marker id="arrAc" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill={color}/></marker></defs>
    <circle cx="150" cy="30" r="22" fill="#F0E4C8" stroke={color} strokeWidth="2" strokeDasharray="3,3"/>
    <text x="150" y="34" textAnchor="middle" fontSize="14" fill={color} fontWeight="800">?</text>
    <circle cx="62" cy="86" r="24" fill="#fff" stroke={BD} strokeWidth="1.5"/>
    <text x="62" y="90" textAnchor="middle" fontSize="11" fill={SB}>age</text>
    <circle cx="238" cy="86" r="24" fill="#fff" stroke={BD} strokeWidth="1.5"/>
    <text x="238" y="90" textAnchor="middle" fontSize="11" fill={SB}>cars owned</text>
    <path d="M 132 42 L 80 70" stroke={color} strokeWidth="2.2" markerEnd="url(#arrAc)"/>
    <path d="M 168 42 L 220 70" stroke={color} strokeWidth="2.2" markerEnd="url(#arrAc)"/>
    <text x="150" y="113" textAnchor="middle" fontSize="9.5" fill={SB}>what unobserved factor could drive both?</text>
  </svg>;

  // randomize: population → random coin → treatment / control
  if(type==="randomize")return <svg viewBox="0 0 300 100" style={s}>
    <defs><marker id="arrRn" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill={color}/></marker></defs>
    <rect x="14" y="36" width="86" height="34" rx="6" fill="#fff" stroke={BD} strokeWidth="1.5"/>
    <text x="57" y="56" textAnchor="middle" fontSize="11" fill={SB}>population</text>
    <circle cx="124" cy="53" r="11" fill={color} opacity=".25" stroke={color}/>
    <text x="124" y="57" textAnchor="middle" fontSize="13" fill={color} fontWeight="800">?</text>
    <path d="M 138 49 L 180 30" stroke={color} strokeWidth="2.2" markerEnd="url(#arrRn)"/>
    <path d="M 138 58 L 180 78" stroke={color} strokeWidth="2.2" markerEnd="url(#arrRn)"/>
    <rect x="184" y="14" width="100" height="34" rx="6" fill={color} opacity=".18" stroke={color} strokeWidth="1.5"/>
    <text x="234" y="34" textAnchor="middle" fontSize="11" fill={color} fontWeight="700">treatment</text>
    <rect x="184" y="60" width="100" height="34" rx="6" fill={SB} opacity=".15" stroke={SB} strokeWidth="1.5"/>
    <text x="234" y="80" textAnchor="middle" fontSize="11" fill={SB} fontWeight="700">control</text>
  </svg>;

  // domain_knowledge: book/brain feeding into a DAG
  if(type==="domain_knowledge")return <svg viewBox="0 0 300 110" style={s}>
    <defs><marker id="arrDk" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill={SM}/></marker></defs>
    <rect x="20" y="32" width="56" height="44" rx="4" fill={color} opacity=".18" stroke={color}/>
    {[44,52,60,68].map(y=><line key={y} x1="30" y1={y} x2={y===68?60:64} y2={y} stroke={color} strokeWidth="1.6"/>)}
    <text x="48" y="92" textAnchor="middle" fontSize="9.5" fill={color} fontWeight="700">domain</text>
    <text x="48" y="103" textAnchor="middle" fontSize="9.5" fill={color} fontWeight="700">knowledge</text>
    <path d="M 84 54 L 122 54" stroke={SM} strokeWidth="2.2" markerEnd="url(#arrDk)"/>
    <circle cx="186" cy="32" r="16" fill="#F0E4C8" stroke={color}/>
    <text x="186" y="36" textAnchor="middle" fontSize="10" fill={color} fontWeight="700">Z?</text>
    <circle cx="160" cy="74" r="14" fill="#fff" stroke={BD}/>
    <circle cx="218" cy="74" r="14" fill="#fff" stroke={BD}/>
    <text x="160" y="78" textAnchor="middle" fontSize="10" fill={SB}>X</text>
    <text x="218" y="78" textAnchor="middle" fontSize="10" fill={SB}>Y</text>
    <line x1="176" y1="44" x2="166" y2="62" stroke={color} strokeWidth="1.6"/>
    <line x1="196" y1="44" x2="212" y2="62" stroke={color} strokeWidth="1.6"/>
    <line x1="172" y1="74" x2="206" y2="74" stroke={SB} strokeWidth="1.5" strokeDasharray="3,3"/>
    <text x="282" y="58" textAnchor="end" fontSize="9.5" fill={SB}>identify</text>
    <text x="282" y="69" textAnchor="end" fontSize="9.5" fill={SB}>confounders</text>
  </svg>;

  // four_clean_plots: 4 mini diagnostic plots that all look fine, with a caveat
  if(type==="four_clean_plots")return <svg viewBox="0 0 300 130" style={s}>
    {[[10,16],[156,16],[10,72],[156,72]].map(([x,y],i)=><g key={i}>
      <rect x={x} y={y} width="134" height="44" fill="none" stroke={BD} rx="3"/>
      <line x1={x+4} y1={y+22} x2={x+130} y2={y+22} stroke={DH} strokeDasharray="2,2" opacity=".4"/>
      {[8,22,38,56,76,94,112,128].map((dx,j)=><circle key={j} cx={x+dx} cy={y+22+([2,-3,1,-2,3,-2,2,-1][j])} r="1.7" fill={color} opacity=".75"/>)}
      <text x={x+6} y={y+10} fontSize="9" fill={SB} fontWeight="700">{["Linearity","Homogeneity","Influence","Q-Q"][i]}</text>
    </g>)}
    <text x="150" y="125" textAnchor="middle" fontSize="11" fill={RF} fontWeight="800">all clean, but did we omit a confounder?</text>
  </svg>;

  return null;
}
function Quiz({questions,color}){
  const[answers,setAnswers]=useState({});
  const[open,setOpen]=useState(false);
  if(!questions||!questions.length) return null;
  const pick=(qi,oi)=>{setAnswers(a=>({...a,[qi]:oi}));};
  const mcqCount=questions.filter(q=>!q.prose).length;
  const score=Object.keys(answers).filter(k=>!questions[k].prose && answers[k]===questions[k].ans).length;
  const mcqAnswered=Object.keys(answers).filter(k=>!questions[k].prose).length;
  const c=color||C.sub;
  return(<div style={{marginTop:18,borderTop:`1px solid ${C.border}`,paddingTop:14}}>
    <button onClick={()=>setOpen(!open)} style={{display:"inline-flex",alignItems:"center",gap:9,padding:"12px 22px",borderRadius:999,border:`2px solid ${c}`,background:open?c:`${c}15`,fontSize:14,fontWeight:800,color:open?"#fff":c,cursor:"pointer",fontFamily:"inherit",boxShadow:open?`0 8px 20px ${c}30`:`0 3px 10px ${c}20`,letterSpacing:".2px",transition:"all .15s"}}>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" style={{flex:"0 0 auto"}}>
        <path d="M9.5 9a2.5 2.5 0 0 1 5 0c0 1.5-1.5 2-2.5 3v1"/>
        <circle cx="12" cy="17" r=".6" fill="currentColor"/>
        <circle cx="12" cy="12" r="9.5"/>
      </svg>
      <span>{open?"Hide":"Show"} Practice Questions{mcqAnswered>0?` · ${score}/${mcqCount}`:` · ${questions.length} Q`}</span>
    </button>
    {open&&<div style={{marginTop:12,display:"flex",flexDirection:"column",gap:16}}>
      {questions.map((q,qi)=>{
        const picked=answers[qi];
        const revealed=picked!=null;
        if(q.prose){
          return(<div key={qi} style={{background:C.card,borderRadius:12,padding:"14px 16px",border:`1px solid ${C.border}`,boxShadow:"0 2px 10px rgba(0,0,0,.04)"}}>
            {q.v&&<QuizVisual type={q.v} color={color}/>}
            <div style={{fontSize:13,fontWeight:600,color:C.text,marginBottom:10,lineHeight:1.5}}>{qi+1}. {q.q}</div>
            {!revealed&&<button onClick={()=>pick(qi,"revealed")} style={{padding:"7px 14px",borderRadius:8,border:`1.5px solid ${c}`,background:`${c}15`,color:c,fontSize:12.5,fontWeight:700,cursor:"pointer",fontFamily:"inherit"}}>Show answer</button>}
            {revealed&&<div style={{marginTop:4,background:`${color}10`,borderLeft:`4px solid ${color}`,borderRadius:8,padding:"10px 12px"}}>
              {q.ev&&<QuizVisual type={q.ev} color={color}/>}
              <div style={{fontSize:12.5,color:C.sub,lineHeight:1.55}}>{q.prose}</div>
            </div>}
          </div>);
        }
        const correctText=q.opts[q.ans];
        return(<div key={qi} style={{background:C.card,borderRadius:12,padding:"14px 16px",border:`1px solid ${C.border}`,boxShadow:"0 2px 10px rgba(0,0,0,.04)"}}>
        {q.v&&<QuizVisual type={q.v} color={color}/>}
        <div style={{fontSize:13,fontWeight:600,color:C.text,marginBottom:8,lineHeight:1.5}}>{qi+1}. {q.q}</div>
        <div style={{display:"flex",flexDirection:"column",gap:5}}>
          {q.opts.map((o,oi)=>{
            const chosen=picked===oi;
            const correct=oi===q.ans;
            let bg2="transparent",brd=`1.5px solid ${C.border}`,col=C.text;
            if(revealed&&correct){bg2="#E7F5EF";brd="1.5px solid #2F855A";col="#236B45";}
            else if(chosen&&!correct){bg2="#FDECEC";brd="1.5px solid #C53030";col="#A62626";}
            return(<button key={oi} onClick={()=>{if(picked==null)pick(qi,oi);}} disabled={picked!=null} style={{textAlign:"left",padding:"8px 12px",borderRadius:8,border:brd,background:bg2,fontSize:13,color:col,cursor:picked!=null?"default":"pointer",fontFamily:"inherit",lineHeight:1.4,fontWeight:chosen||(revealed&&correct)?700:500,opacity:revealed&&!correct&&!chosen ? .55 : 1}}>
              {String.fromCharCode(65+oi)}) {o}
            </button>);
          })}
        </div>
        {revealed&&<div style={{marginTop:10,background:`${color}10`,borderLeft:`4px solid ${picked===q.ans?"#2F855A":"#C53030"}`,borderRadius:8,padding:"10px 12px"}}>
          <div style={{fontSize:12,fontWeight:800,color:picked===q.ans?"#236B45":"#A62626",marginBottom:4}}>{picked===q.ans?"Correct":"Not quite"} · Answer: {correctText}</div>
          {q.ev&&<QuizVisual type={q.ev} color={color}/>}
          <div style={{fontSize:12.5,color:C.sub,lineHeight:1.55}}>{q.explain}</div>
        </div>}
      </div>);})}
      {mcqAnswered===mcqCount&&mcqCount>0&&<div style={{fontSize:14,fontWeight:700,color:color||C.text,textAlign:"center",padding:8}}>Score: {score}/{mcqCount}</div>}
    </div>}
  </div>);
}

/* ── DIAG PLOTS ──────────────────────────────────────────────────────── */
function DiagPlot({type,model,highlight,hiIdx,onHi}){
  const brd=highlight?`2.5px solid ${highlight}`:`1.5px solid ${C.border}`;const bg=highlight?`${highlight}08`:"transparent";
  const titles={linearity:"Linearity",homogeneity:"Homogeneity of Variance",influential:"Influential Observations",normality:"Normality of Residuals"};
  const subtitles={linearity:"Reference line should be flat and horizontal",homogeneity:"Reference line should be flat and horizontal",influential:"Points should be inside the contour lines",normality:"Dots should fall along the line"};
  if(!model)return <div style={{border:brd,borderRadius:10,padding:10,background:bg,display:"flex",alignItems:"center",justifyContent:"center",minHeight:200}}><span style={{fontSize:13,color:C.muted}}>Need 3+ points</span></div>;
  const{fitted,residuals,stdRes,cooks,n}=model;
  const dot=(cx,cy,idx)=>{const isH=hiIdx===idx;return <g key={idx} style={{cursor:"pointer"}} onClick={e=>{e.stopPropagation();onHi?.(hiIdx===idx?null:idx);}}>{isH&&<circle cx={cx} cy={cy} r={9} fill={C.hiG}/>}<circle cx={cx} cy={cy} r={isH?5.5:3.5} fill={isH?C.hi:C.dot} stroke={isH?"#C53030":C.dotS} strokeWidth={isH?1.5:.6} opacity={(hiIdx!=null&&!isH) ? .3 : .8}/>{isH&&<text x={cx+8} y={cy-5} fontSize={9} fill={C.hi} fontWeight={700}>#{idx+1}</text>}</g>;};
  const wrap=(ch)=><div style={{border:brd,borderRadius:12,padding:"9px 9px 3px",background:highlight?bg:C.card,boxShadow:highlight?`0 0 0 3px ${highlight}12, 0 6px 18px rgba(0,0,0,.05)`:"0 2px 10px rgba(0,0,0,.04)",transition:"all .2s"}} onClick={()=>onHi?.(null)}><div style={{fontSize:13,fontWeight:800,color:C.text,paddingLeft:4,lineHeight:1.15}}>{titles[type]}</div><div style={{fontSize:10.5,color:C.sub,paddingLeft:4,marginBottom:2,lineHeight:1.15}}>{subtitles[type]}</div>{ch}</div>;
  const ciBand=(sm,sx,sy)=>{if(sm.length<2)return null;const top=sm.map(p=>`L${sx(p.x)},${sy(p.hi)}`).join(" ");const bot=[...sm].reverse().map(p=>`L${sx(p.x)},${sy(p.lo)}`).join(" ");return <path d={`M${sx(sm[0].x)},${sy(sm[0].hi)} ${top.slice(1)} ${bot} Z`} fill={C.smooth} opacity={.13} stroke="none"/>;};
  const ciLine=(sm,sx,sy)=>sm.length>1?<path d={sm.map((p,i)=>`${i?'L':'M'}${sx(p.x)},${sy(p.y)}`).join(' ')} fill="none" stroke={C.smooth} strokeWidth={2.2} opacity={.9}/>:null;
  if(type==="linearity"){const sm=loess(fitted,residuals);const ys=[...residuals,...sm.map(p=>p.lo),...sm.map(p=>p.hi)];return wrap(<CF xs={fitted} ys={ys} xL="Fitted values" yL="Residuals">{(sx,sy,w)=><>{<line x1={0} x2={w} y1={sy(0)} y2={sy(0)} stroke={C.dash} strokeWidth={1} strokeDasharray="5,5" opacity={.65}/>}{ciBand(sm,sx,sy)}{ciLine(sm,sx,sy)}{fitted.map((f,i)=>dot(sx(f),sy(residuals[i]),i))}</>}</CF>);}
  if(type==="homogeneity"){const sqA=stdRes.map(r=>Math.sqrt(Math.abs(r)));const sm=loess(fitted,sqA);const ys=[...sqA,...sm.map(p=>p.lo),...sm.map(p=>p.hi)];return wrap(<CF xs={fitted} ys={ys} xL="Fitted values" yL={"\u221A|Std. residuals|"}>{(sx,sy)=><>{ciBand(sm,sx,sy)}{ciLine(sm,sx,sy)}{fitted.map((f,i)=>dot(sx(f),sy(sqA[i]),i))}</>}</CF>);}
  if(type==="influential"){
    const p=2,hat=model.hat||[],maxH=Math.max(.08,...hat)*1.18;
    const df2=Math.max(n-p,1);
    const qt75=0.67449+0.2454/df2+0.0795/(df2*df2);
    const threshold=qt75*qt75;
    const thresholdLabel=threshold.toFixed(1);
    const sm=loess(hat,stdRes);
    const yCap=5;
    const contourYs=[];
    for(let i=1;i<=80;i++){const hv=(maxH*.98)*i/80;const rv=Math.sqrt(Math.max(0,threshold*p*(1-hv)/hv));if(Number.isFinite(rv)&&rv<=yCap){contourYs.push(rv);contourYs.push(-rv);}}
    const xs=[0,...hat,maxH],ys=[-3,3,...stdRes,...sm.map(s=>s.y),...contourYs];
    return wrap(<CF xs={xs} ys={ys} xL="Leverage" yL="Std. Residuals">{(sx,sy,w,h)=>{
      const showLabel=new Set(cooks.map((c,i)=>({c,i})).sort((a,b)=>b.c-a.c).slice(0,Math.max(3,cooks.filter(c=>c>=threshold).length)).map(d=>d.i));
      const curve=(D,sgn)=>{const pts=[];for(let i=1;i<=80;i++){const hv=(maxH*.98)*i/80;const rv=sgn*Math.sqrt(Math.max(0,D*p*(1-hv)/hv));if(Number.isFinite(rv)&&Math.abs(rv)<=yCap)pts.push([sx(hv),sy(rv)]);}return pts.length>1?<path key={`${D}-${sgn}`} d={pts.map((pt,i)=>`${i?"L":"M"}${pt[0]},${pt[1]}`).join(" ")} fill="none" stroke={C.smooth} strokeWidth={1.4} strokeDasharray="7,6" opacity={.75}/>:null;};
      const minH=Math.min(...hat,maxH/3);
      const hat80=minH+(maxH-minH)*0.8;
      const labelHv=Math.min(hat80,maxH*0.85);
      const labelRv=Math.sqrt(Math.max(0,threshold*p*(1-labelHv)/labelHv));
      const lp=Number.isFinite(labelRv)&&labelRv>0.2&&labelRv<yCap-0.2?{hv:labelHv,rv:labelRv}:null;
      const smPath=sm.length>1?<path d={sm.map((pt,i)=>`${i?'L':'M'}${sx(pt.x)},${sy(pt.y)}`).join(' ')} fill="none" stroke={C.smooth} strokeWidth={2} opacity={.9}/>:null;
      return <>{<line x1={0} x2={w} y1={sy(0)} y2={sy(0)} stroke={C.dash} strokeWidth={1} strokeDasharray="5,5" opacity={.35}/>}
        {curve(threshold,1)}{curve(threshold,-1)}
        {lp&&<text x={sx(lp.hv)-4} y={sy(lp.rv)-7} fontSize={10} fill={C.smooth} fontWeight={700} opacity={.9} textAnchor="end">{`Cook's D = ${thresholdLabel}`}</text>}
        {smPath}
        {hat.map((hv,i)=>{const isH=hiIdx===i;const ov=cooks[i]>=threshold;const lab=isH||showLabel.has(i);return <g key={i} style={{cursor:"pointer"}} onClick={e=>{e.stopPropagation();onHi?.(hiIdx===i?null:i);}}>{isH&&<circle cx={sx(hv)} cy={sy(stdRes[i])} r={10} fill={C.hiG}/>}<circle cx={sx(hv)} cy={sy(stdRes[i])} r={isH?5.5:3.4} fill={isH?C.hi:ov?C.ref:C.dot} stroke={isH?"#C53030":ov?C.ref:C.dotS} strokeWidth={isH?1.5:.7} opacity={(hiIdx!=null&&!isH) ? .28 : .82}/>{lab&&<text x={sx(hv)+7} y={sy(stdRes[i])-4} fontSize={9} fill={isH?C.hi:C.dotS} fontWeight={700}>{i+1}</text>}</g>;})}
      </>;
    }}</CF>);
  }
  if(type==="normality"){
    const qq=qqWorm(model.studentRes||stdRes);
    const ys=[...qq.map(q=>q.dev),...qq.map(q=>q.lo),...qq.map(q=>q.hi),0];
    return wrap(<CF xs={qq.map(q=>q.th)} ys={ys} xL="Standard Normal Distribution Quantiles" yL="Sample Quantile Deviations">{(sx,sy,w)=>{
      const top=qq.map(q=>`L${sx(q.th)},${sy(q.hi)}`).join(" ");
      const bot=[...qq].reverse().map(q=>`L${sx(q.th)},${sy(q.lo)}`).join(" ");
      const band=qq.length>1?<path d={`M${sx(qq[0].th)},${sy(qq[0].hi)} ${top.slice(1)} ${bot} Z`} fill={C.smooth} opacity={.13} stroke="none"/>:null;
      return <>{band}<line x1={0} x2={w} y1={sy(0)} y2={sy(0)} stroke={C.smooth} strokeWidth={2} opacity={.85}/>{qq.map(q=>dot(sx(q.th),sy(q.dev),q.oi))}</>;
    }}</CF>);
  }
  return null;
}

/* ── SCATTER ─────────────────────────────────────────────────────────── */
function Scatter({points,setPoints,model,hiIdx,onHi,onEdit}){
  const ref=useRef(null);const[drag,setDrag]=useState(null);const moved=useRef(false);const lockedScale=useRef(null);
  const W=460,H=340,pad={top:22,right:16,bottom:42,left:50},pw=W-pad.left-pad.right,ph=H-pad.top-pad.bottom;
  const xs=points.map(p=>p.x),ys=points.map(p=>p.y);
  const liveSxx=sc(xs.length?xs:[0,10],pw),liveSyy=sc(ys.length?ys:[0,10],ph);
  const sxx=drag!==null&&lockedScale.current?lockedScale.current.sxx:liveSxx;
  const syy=drag!==null&&lockedScale.current?lockedScale.current.syy:liveSyy;
  const toD=useCallback((cx,cy)=>{const s=ref.current;if(!s)return{x:0,y:0};const r=s.getBoundingClientRect();const sx_=(cx-r.left)/r.width*W-pad.left,sy_=(cy-r.top)/r.height*H-pad.top;return{x:+(sxx.min+(sx_/pw)*(sxx.max-sxx.min)).toFixed(2),y:+(syy.min+((ph-sy_)/ph)*(syy.max-syy.min)).toFixed(2)};},[sxx.min,sxx.max,syy.min,syy.max,pw,ph]);
  return <svg ref={ref} viewBox={`0 0 ${W} ${H}`} style={{width:"100%",height:"auto",cursor:"crosshair",display:"block"}}
    onDoubleClick={e=>{if(!e.target.closest("circle")){const d=toD(e.clientX,e.clientY);setPoints(p=>[...p,d]);onEdit?.();}}}
    onPointerMove={e=>{if(drag!==null){moved.current=true;const d=toD(e.clientX,e.clientY);setPoints(p=>p.map((pt,i)=>i===drag?d:pt));}}}
    onPointerUp={()=>{if(drag!==null&&moved.current)onEdit?.();setDrag(null);lockedScale.current=null;moved.current=false;}}
    onPointerCancel={()=>{setDrag(null);lockedScale.current=null;moved.current=false;}}
    onClick={e=>{if(drag===null&&e.detail===1&&!e.target.closest("circle"))onHi?.(null);}}
  ><g transform={`translate(${pad.left},${pad.top})`}>
    {sxx.ticks.map(t=><line key={t} x1={sxx.fn(t)} x2={sxx.fn(t)} y1={0} y2={ph} stroke={C.grid} strokeWidth={.6}/>)}
    {syy.ticks.map(t=><line key={t} x1={0} x2={pw} y1={ph-syy.fn(t)} y2={ph-syy.fn(t)} stroke={C.grid} strokeWidth={.6}/>)}
    <line x1={0} x2={pw} y1={ph} y2={ph} stroke={C.border}/><line x1={0} x2={0} y1={0} y2={ph} stroke={C.border}/>
    {sxx.ticks.map(t=><text key={`l${t}`} x={sxx.fn(t)} y={ph+17} textAnchor="middle" fontSize={11} fill={C.sub}>{+t.toFixed(1)}</text>)}
    {syy.ticks.map(t=><text key={`l${t}`} x={-8} y={ph-syy.fn(t)+4} textAnchor="end" fontSize={11} fill={C.sub}>{+t.toFixed(1)}</text>)}
    <text x={pw/2} y={ph+36} textAnchor="middle" fontSize={12} fill={C.text} fontWeight={600}>X</text>
    <text x={-38} y={ph/2} textAnchor="middle" fontSize={12} fill={C.text} fontWeight={600} transform={`rotate(-90,-38,${ph/2})`}>Y</text>
    {model&&<line x1={sxx.fn(sxx.min)} y1={ph-syy.fn(model.b0+model.b1*sxx.min)} x2={sxx.fn(sxx.max)} y2={ph-syy.fn(model.b0+model.b1*sxx.max)} stroke={C.ref} strokeWidth={2.5} opacity={.55}/>}
    {points.map((p,i)=>{const isH=hiIdx===i;return <g key={i}>{isH&&<circle cx={sxx.fn(p.x)} cy={ph-syy.fn(p.y)} r={12} fill={C.hiG}/>}<circle cx={sxx.fn(p.x)} cy={ph-syy.fn(p.y)} r={drag===i?7:isH?6.5:5} fill={isH?C.hi:C.dot} stroke={isH?"#C53030":drag===i?C.ref:C.dotS} strokeWidth={isH||drag===i?2:1} opacity={(hiIdx!=null&&!isH) ? .25 : .85} style={{cursor:"grab"}} onPointerDown={e=>{e.stopPropagation();e.target.setPointerCapture(e.pointerId);lockedScale.current={sxx:liveSxx,syy:liveSyy};setDrag(i);moved.current=false;}} onClick={e=>{e.stopPropagation();if(!moved.current)onHi?.(hiIdx===i?null:i);}} onDoubleClick={e=>{e.stopPropagation();e.preventDefault();setPoints(pp=>pp.filter((_,j)=>j!==i));onEdit?.();}}/>{isH&&<text x={sxx.fn(p.x)+9} y={ph-syy.fn(p.y)-7} fontSize={11} fill={C.hi} fontWeight={700}>#{i+1}</text>}</g>;})}
  </g></svg>;
}

/* ── AUTOCORRELATION DIAGRAMS ────────────────────────────────────────── */
function AutocorrDiags(){
  // 20 weeks of flu surveillance. Flu has a single peak around weeks 8..10 that happens
  // to coincide with the warmest weeks. The naive scatter looks like a clean positive
  // relationship. The time view reveals the peak is one short outbreak whose timing
  // overlaps with the warm spell, so the apparent temperature effect is largely a
  // co-trend rather than a causal driver.
  const data=[
    {w:1,t:8,f:26},{w:2,t:11,f:32},{w:3,t:9,f:28},{w:4,t:13,f:38},{w:5,t:16,f:48},
    {w:6,t:19,f:58},{w:7,t:22,f:70},{w:8,t:24,f:78},{w:9,t:25,f:82},{w:10,t:23,f:74},
    {w:11,t:20,f:62},{w:12,t:17,f:52},{w:13,t:14,f:42},{w:14,t:11,f:34},{w:15,t:9,f:28},
    {w:16,t:7,f:24},{w:17,t:8,f:26},{w:18,t:10,f:30},{w:19,t:6,f:22},{w:20,t:9,f:28}
  ];
  // Left panel: temperature 6..25 -> x 24..192; flu 22..82 -> y 96..18
  const tx=t=>24+((t-6)/19)*168;
  const fy=f=>96-((f-22)/60)*78;
  // Right panel: week 1..20 -> x 24..186; flu uses same fy mapping
  const wx=w=>24+((w-1)/19)*162;
  const tsPath=data.map((p,i)=>`${i?"L":"M"}${wx(p.w)},${fy(p.f)}`).join(" ");
  return <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12,margin:"12px 0"}}>
    <div style={{background:C.bg,borderRadius:10,padding:12,border:`1.5px solid ${C.border}`}}>
      <div style={{fontSize:12,fontWeight:700,color:"#2E86AB",marginBottom:6}}>Flu cases vs. temperature (looks fine)</div>
      <svg viewBox="0 0 200 120" style={{width:"100%",height:"auto"}}>
        <line x1="24" x2="24" y1="18" y2="98" stroke={C.border} strokeWidth=".8"/>
        <line x1="24" x2="194" y1="98" y2="98" stroke={C.border} strokeWidth=".8"/>
        <line x1={tx(6)} y1={fy(22)} x2={tx(25)} y2={fy(82)} stroke="#2E86AB" strokeWidth="1.6" opacity=".4"/>
        {data.map(p=><g key={p.w}>
          <circle cx={tx(p.t)} cy={fy(p.f)} r="6" fill="#fff" stroke="#2E86AB" strokeWidth="1.4"/>
          <text x={tx(p.t)} y={fy(p.f)+2.7} textAnchor="middle" fontSize="7.2" fill="#2E86AB" fontWeight="700">{p.w}</text>
        </g>)}
        <text x="112" y="115" textAnchor="middle" fontSize="10" fill={C.sub}>Temperature</text>
        <text x="10" y="60" textAnchor="middle" fontSize="10" fill={C.sub} transform="rotate(-90,10,60)">Flu cases</text>
      </svg>
      <div style={{fontSize:11,color:C.sub,lineHeight:1.5,marginTop:4}}>Warmer weeks have more flu cases, with a clean positive slope. Each circle is a single week, labelled by week number.</div>
    </div>
    <div style={{background:"#FFF5F5",borderRadius:10,padding:12,border:"1.5px solid #FED7D7"}}>
      <div style={{fontSize:12,fontWeight:700,color:"#C53030",marginBottom:6}}>Same flu cases, plotted by week</div>
      <svg viewBox="0 0 200 120" style={{width:"100%",height:"auto"}}>
        <line x1="18" x2="18" y1="18" y2="98" stroke={C.border} strokeWidth=".8"/>
        <line x1="18" x2="192" y1="98" y2="98" stroke={C.border} strokeWidth=".8"/>
        <path d={tsPath} fill="none" stroke="#C53030" strokeWidth="1.2" opacity=".55"/>
        {data.map(p=><g key={p.w}>
          <circle cx={wx(p.w)} cy={fy(p.f)} r="5.4" fill="#fff" stroke="#C53030" strokeWidth="1.4"/>
          <text x={wx(p.w)} y={fy(p.f)+2.5} textAnchor="middle" fontSize="6.8" fill="#C53030" fontWeight="700">{p.w}</text>
        </g>)}
        <text x="105" y="115" textAnchor="middle" fontSize="10" fill={C.sub}>Week number</text>
        <text x="8" y="60" textAnchor="middle" fontSize="10" fill={C.sub} transform="rotate(-90,8,60)">Flu cases</text>
      </svg>
      <div style={{fontSize:11,color:"#C53030",lineHeight:1.5,marginTop:4,fontWeight:600}}>The whole positive relationship comes from a single outbreak peak around weeks 8 to 10. That peak happens to overlap with the warmest weeks, so the regression treats warmth as the driver. Adjacent weeks share unmodelled outbreak dynamics, so the observations are not independent. The same week labels appear in both panels for tracking.</div>
    </div>
  </div>;
}

function ClusteredDiags(){
  const[showColors,setShowColors]=useState(false);
  const clinics=[
    {name:"A",x:34,offset:-12,c:"#2E86AB"},{name:"B",x:70,offset:11,c:"#6B46C1"},
    {name:"C",x:106,offset:-3,c:"#2F855A"},{name:"D",x:142,offset:14,c:"#C53030"},
    {name:"E",x:178,offset:-8,c:"#8B6914"}
  ];
  const noise=[-9,6,13,-5,8];
  const pts=clinics.flatMap((cl,ci)=>[0,1,2,3,4].map(j=>{
    const x=34+j*34+(ci-2)*3;
    const y=90-.24*x+cl.offset+noise[(j+ci)%noise.length];
    return{cl:cl.name,c:cl.c,x,y,rx:cl.x+(j-2)*4,ry:60+cl.offset+noise[(j+ci)%noise.length]*.5};
  }));
  const pointFill=p=>showColors?p.c:"#6B46C1";
  return <div>
    <button onClick={()=>setShowColors(v=>!v)} style={{margin:"8px 0 10px",padding:"7px 12px",borderRadius:999,border:"1.5px solid #6B46C1",background:showColors?"#6B46C1":"#fff",color:showColors?"#fff":"#6B46C1",fontSize:12,fontWeight:800,fontFamily:"inherit",cursor:"pointer"}}>
      {showColors?"Hide":"Reveal"} clinic colors
    </button>
    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12,margin:"0 0 12px"}}>
    <div style={{background:C.bg,borderRadius:10,padding:12,border:`1.5px solid ${C.border}`}}>
      <div style={{fontSize:12,fontWeight:700,color:"#6B46C1",marginBottom:6}}>Naive scatter: 25 dots look independent</div>
      <svg viewBox="0 0 210 130" style={{width:"100%",height:"auto"}}>
        <line x1="28" x2="28" y1="15" y2="104" stroke={C.border}/><line x1="28" x2="195" y1="104" y2="104" stroke={C.border}/>
        <line x1="34" x2="185" y1="82" y2="46" stroke="#6B46C1" strokeWidth="2" opacity=".35"/>
        {pts.map((p,i)=><circle key={i} cx={p.x} cy={p.y} r="3.5" fill={pointFill(p)} opacity=".7"/>)}
        <text x="110" y="123" textAnchor="middle" fontSize="10" fill={C.sub}>Minutes with provider</text>
        <text x="9" y="60" textAnchor="middle" fontSize="10" fill={C.sub} transform="rotate(-90,9,60)">Satisfaction</text>
      </svg>
      <div style={{fontSize:11,color:C.sub,lineHeight:1.5,marginTop:4}}>Ignoring clinic makes this look like 25 separate patients.</div>
    </div>
    <div style={{background:"#F5F0FF",borderRadius:10,padding:12,border:"1.5px solid #D0C0E8"}}>
      <div style={{fontSize:12,fontWeight:700,color:"#6B46C1",marginBottom:6}}>Same data grouped by clinic</div>
      <svg viewBox="0 0 210 130" style={{width:"100%",height:"auto"}}>
        <line x1="28" x2="28" y1="15" y2="104" stroke={C.border}/><line x1="28" x2="195" y1="104" y2="104" stroke={C.border}/>
        {clinics.map(cl=><g key={cl.name}><rect x={cl.x-13} y={18} width={26} height={84} rx="6" fill={showColors?cl.c:"#6B46C1"} opacity=".08" stroke={showColors?cl.c:"#6B46C1"} strokeOpacity=".28"/><text x={cl.x} y="116" textAnchor="middle" fontSize="10" fill={showColors?cl.c:C.sub} fontWeight="700">{cl.name}</text></g>)}
        {pts.map((p,i)=><circle key={i} cx={p.rx} cy={p.ry} r="3.4" fill={pointFill(p)} opacity=".76"/>)}
        <text x="9" y="60" textAnchor="middle" fontSize="10" fill={C.sub} transform="rotate(-90,9,60)">Satisfaction</text>
      </svg>
      <div style={{fontSize:11,color:"#6B46C1",lineHeight:1.5,marginTop:4,fontWeight:600}}>Satisfaction values cluster within clinic. The effective information is closer to 5 clinics than 25 independent patients.</div>
    </div>
    </div>
  </div>;
}

/* ── INFO PANELS ─────────────────────────────────────────────────────── */
const ul={margin:0,paddingLeft:20,fontSize:14,lineHeight:1.75,color:C.text};
const pa={margin:0,fontSize:14,lineHeight:1.75,color:C.text};

function MdBold({text}){if(!text)return null;return text.split(/(\*\*[^*]+\*\*)/g).map((p,i)=>p.startsWith("**")&&p.endsWith("**")?<b key={i}>{p.slice(2,-2)}</b>:<span key={i}>{p}</span>);}

function pointsToCSV(points){const rows=["x,y",...points.map(p=>`${p.x},${p.y}`)];return rows.join("\n")+"\n";}
function pointsToR(points,name){
  const xs=points.map(p=>p.x).join(", ");
  const ys=points.map(p=>p.y).join(", ");
  const safe=(name||"regression_data").replace(/[^a-zA-Z0-9_]/g,"_").replace(/^_+|_+$/g,"")||"regression_data";
  return [
    "# Install once (qqplotr is needed for the Q-Q confidence band)",
    "pkgs <- c(\"performance\", \"see\", \"qqplotr\")",
    "to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]",
    "if (length(to_install)) install.packages(to_install)",
    "",
    "library(performance)",
    "library(see)",
    "",
    "# Data from the explorer",
    `${safe} <- data.frame(`,
    `  x = c(${xs}),`,
    `  y = c(${ys})`,
    ")",
    "",
    "# Fit linear regression",
    `model <- lm(y ~ x, data = ${safe})`,
    "summary(model)",
    "",
    "# All diagnostic plots in one panel (mirrors this explorer)",
    "check_model(model)",
    ""
  ].join("\n");
}
function downloadText(filename,text,mime="text/plain"){
  const blob=new Blob([text],{type:`${mime};charset=utf-8`});
  const url=URL.createObjectURL(blob);
  const a=document.createElement("a");
  a.href=url;a.download=filename;document.body.appendChild(a);a.click();
  setTimeout(()=>{document.body.removeChild(a);URL.revokeObjectURL(url);},0);
}
async function copyText(text){
  try{if(navigator.clipboard&&navigator.clipboard.writeText){await navigator.clipboard.writeText(text);return true;}}catch(_){}
  try{const ta=document.createElement("textarea");ta.value=text;ta.style.position="fixed";ta.style.opacity="0";document.body.appendChild(ta);ta.select();const ok=document.execCommand("copy");document.body.removeChild(ta);return ok;}catch(_){return false;}
}

const INFO={
sample:()=><div style={{display:"flex",flexDirection:"column",gap:16}}>
  <div><Hd>What it means</Hd><p style={pa}>You need enough observations for regression estimates to be stable. Too few, and the slope bounces around unpredictably from sample to sample.</p></div>
  <div><Hd>Rules of thumb</Hd><ul style={ul}>
    <li><b>Simple linear regression:</b> around <b>20 observations</b> is a common minimum starting point, and 30+ is often more comfortable. Treat this as guidance, not a magic cutoff.</li>
    <li><b>Multiple regression:</b> a common starting point is <b>10 to 15 observations per predictor</b>. Tabachnick and Fidell (2019) recommend <b>N &ge; 50 + 8k</b> (k = number of predictors) for testing the overall model. Green (1991) suggests <b>N &ge; 104 + k</b> for individual predictors.</li>
    <li><b>Effect size matters:</b> strong relationships need fewer observations. Subtle effects need more.</li>
  </ul></div>
  <div><Hd>What goes wrong with too few</Hd><ul style={ul}>
    <li>Coefficient estimates become <b>unstable</b> across different samples.</li>
    <li>Confidence intervals become very <b>wide</b>, making it hard to draw conclusions.</li>
    <li>The model may <b>overfit</b>: capturing noise rather than real patterns.</li>
  </ul></div>
  <div style={{background:"#F5F3EE",borderRadius:8,padding:"10px 14px"}}>
    <div style={{fontSize:11,fontWeight:700,color:C.muted,letterSpacing:.5,marginBottom:4}}>REFERENCES</div>
    <ul style={{...ul,fontSize:12,color:C.sub}}>
      <li>Field, A. (2018). <i>Discovering Statistics Using IBM SPSS Statistics</i> (5th ed.). Sage. <Lnk href="https://us.sagepub.com/en-us/nam/discovering-statistics-using-ibm-spss-statistics/book257672">Publisher</Lnk></li>
      <li>Tabachnick, B. G., & Fidell, L. S. (2019). <i>Using Multivariate Statistics</i> (7th ed.). Pearson. <Lnk href="https://www.pearson.com/en-us/subject-catalog/p/using-multivariate-statistics/P200000003097">Publisher</Lnk></li>
      <li>Green, S. B. (1991). How many subjects does it take to do a regression analysis? <i>Multivariate Behavioral Research, 26</i>(3), 499-510. <Lnk href="https://doi.org/10.1207/s15327906mbr2603_7">DOI</Lnk></li>
    </ul>
  </div>
</div>,

uncorrelated:()=><div style={{display:"flex",flexDirection:"column",gap:18}}>
  <div><Hd>What it means</Hd><p style={pa}>Each observation's prediction error should be independent of every other. Knowing that one residual is positive should tell you nothing about the next one.</p></div>

  <div style={{background:"#F5F0FF",border:"1.5px solid #D0C0E8",borderRadius:12,padding:16,display:"flex",flexDirection:"column",gap:14}}>
    <div style={{fontSize:16,fontWeight:700,color:"#6B46C1"}}>Violation type 1: Clustered data</div>
    <div><Hd>What is happening</Hd><p style={pa}>Observations within the same group (same hospital, same school, same neighborhood) tend to be more alike than observations from other groups. The errors within a cluster are correlated.</p>
    <p style={{...pa,marginTop:6}}><b>Example:</b> you regress patient satisfaction on minutes spent with the provider, with patients drawn from 10 clinics. Patients at the same clinic share the same staff, scheduling system, and crowding level. Their residuals will be correlated.</p></div>
    <ClusteredDiags/>
    <div><Hd>How to assess</Hd><ul style={ul}><li><b>Think about data collection.</b> Are observations grouped by site, school, provider, family?</li><li>Compute the <b>intraclass correlation coefficient (ICC)</b> to measure how much variance is between vs. within clusters.</li></ul></div>
    <div><Hd>What goes wrong</Hd><p style={pa}>Standard errors tend to be too small and p-values too optimistic. The effective sample size is smaller than it looks.</p></div>
    <div><Hd>How to fix</Hd><ul style={ul}><li>Use <b>multilevel/mixed-effects models</b> that account for grouping.</li><li>Use <b>cluster-robust standard errors</b> as a simpler option.</li><li>At minimum, <b>acknowledge</b> the clustering as a limitation.</li></ul></div>
  </div>

  <div style={{background:"#F0F8FF",border:"1.5px solid #B8D8F0",borderRadius:12,padding:16,display:"flex",flexDirection:"column",gap:14}}>
    <div style={{fontSize:16,fontWeight:700,color:"#2E86AB"}}>Violation type 2: Autocorrelation (time-ordered data)</div>
    <div><Hd>What is happening</Hd><p style={pa}>When data is collected over time, nearby observations often have similar residuals. Errors form runs or waves instead of bouncing randomly.</p>
    <p style={{...pa,marginTop:6}}><b>Example:</b> you regress weekly flu cases on temperature. The scatter shows a clean positive slope, suggesting warmer weeks bring more flu. Plotting flu over time tells a different story: a single outbreak happened to peak during the warmest weeks. The naive regression reads that coincidence as a temperature effect.</p></div>
    <AutocorrDiags/>
    <div><Hd>How to assess</Hd><ul style={ul}><li><b>Plot residuals in collection order</b> (by time, by sequence). Look for runs or waves.</li><li><b>Durbin-Watson test</b> checks for first-order autocorrelation.</li></ul></div>
    <div><Hd>What goes wrong</Hd><ul style={ul}><li><b>Standard errors tend to be too small.</b> The model overestimates how much independent information the data contain.</li><li><b>Confidence intervals are typically too narrow</b> and <b>p-values typically too small.</b></li></ul></div>
    <div><Hd>How to fix</Hd><ul style={ul}><li>Use a <b>time-series model</b> (e.g., ARIMA, GLS with autocorrelated errors).</li><li>Add <b>lagged variables</b> as predictors.</li><li>Use <b>Newey-West standard errors</b>.</li></ul></div>
  </div>
</div>,

multicollinearity:()=><div style={{display:"flex",flexDirection:"column",gap:14}}>
  <div style={{background:"#FFF8E1",border:"1.5px solid #E8D080",borderRadius:8,padding:"12px 16px",fontSize:14,color:"#6B5B1F",lineHeight:1.6}}>
    This applies to <b>multiple regression only</b>. With one predictor, it cannot occur. Included here for the mnemonic.
  </div>
  <div><Hd>What it means</Hd><p style={pa}>When two or more predictors are highly correlated with each other, the model struggles to separate their individual effects.</p></div>
  <div><Hd>Two types</Hd><ul style={ul}>
    <li><b>Perfect multicollinearity:</b> one predictor is an exact linear function of another (e.g., temperature in Celsius and in Fahrenheit). The model cannot be estimated at all. Software will drop one variable or throw an error.</li>
    <li><b>Imperfect multicollinearity</b> (the practical concern): predictors are highly but not perfectly correlated. For instance, <b>years of education</b> and <b>household income</b> in a model predicting health outcomes. The model can be estimated, but coefficients become unstable and standard errors inflate.</li>
  </ul></div>
  <div><Hd>How to assess</Hd><ul style={ul}><li><b>VIF (Variance Inflation Factor)</b>. Above 5 or 10 is commonly treated as a problem.</li><li><b>Correlation matrix</b> of predictors.</li></ul></div>
</div>,

exogeneity:()=><div style={{display:"flex",flexDirection:"column",gap:14}}>
  <div><Hd>What it means</Hd><p style={pa}>Your predictor must not be tangled up with the error term. There should be no lurking variable that simultaneously drives both X and Y.</p></div>
  <div><Hd>Example</Hd><p style={pa}>You regress <b>pain scores</b> on <b>physical therapy visits</b> and find a positive slope. The problem: <b>injury severity</b> can drive both. It sits in the error term and is correlated with the predictor, so the coefficient is biased.</p></div>
  <div><Hd>What goes wrong</Hd><p style={pa}>The slope estimate is <b>biased</b>, so it no longer cleanly measures the effect you intend to estimate. More data alone does not fix this.</p></div>
  <div><Hd>How to assess</Hd><p style={pa}>There is no diagnostic plot for this. It is about <b>research design and causal reasoning</b>.</p></div>
  <div><Hd>How to fix</Hd><ul style={ul}>
    <li><b>Randomized experiments</b> solve it by design.</li>
    <li><b>Add control variables</b> in multiple regression.</li>
    <li><b>Instrumental variables</b> in some observational settings.</li>
    <li>At minimum: ask whether a third variable drives both X and Y.</li>
  </ul></div>
  <p style={{...pa,marginTop:4,color:C.muted,fontStyle:"italic"}}>We will return to this with multiple regression and causal inference.</p>
</div>,
};

function InfoPanel({item}){const Content=INFO[item.key];return <div style={{"--accent":item.color,background:C.card,border:`1.5px solid ${C.border}`,borderTop:`7px solid ${item.color}`,borderRadius:16,padding:24,maxWidth:840,boxShadow:C.shadow}}>
  <div style={{fontSize:22,fontWeight:800,color:item.color,marginBottom:2}}>{item.label}</div>
  <div style={{fontSize:14,color:C.sub,marginBottom:16,fontStyle:"italic"}}>{item.summary}</div>
  {Content&&<Content/>}
  <Quiz questions={MCQ[item.key]} color={item.color}/>
</div>;}

/* ── DIAGNOSTIC PANEL ────────────────────────────────────────────────── */
function DiagPanel({item,points,setPoints,model,custom,setCustom}){
  const[exTab,setExTab]=useState("good");const[hiIdx,setHiIdx]=useState(null);const[copied,setCopied]=useState(false);
  const loadEx=t=>{if(custom&&!window.confirm("Loading an example will replace your modified points. Continue?"))return;setExTab(t);setCustom(false);setHiIdx(null);const k=item.examples[t];if(k&&DS[k])setPoints([...DS[k].points]);};
  const onEdit=()=>{setCustom(true);};const active=custom?null:exTab;const ds=active?DS[item.examples[active]]:null;
  const datasetSlug=custom?"regression_data":(item.examples?.[exTab]||"regression_data");
  const handleDownload=()=>{if(!points.length)return;downloadText(`${datasetSlug}.csv`,pointsToCSV(points),"text/csv");};
  const handleCopyR=async()=>{if(!points.length)return;const ok=await copyText(pointsToR(points,datasetSlug));if(ok){setCopied(true);setTimeout(()=>setCopied(false),1500);}};
  const tabOrder=["good","borderline","bad"];const tabIcons={good:"\u2713",borderline:"~",bad:"\u2717"};const tabLabels={good:"Good Fit",borderline:"Borderline",bad:"Clear Violation"};
  const pill=k=>({padding:"8px 16px",borderRadius:20,fontSize:13,fontWeight:active===k?700:500,border:`1.5px solid ${active===k?item.color:C.border}`,background:active===k?item.colorSoft:"transparent",color:active===k?item.color:C.muted,cursor:"pointer",transition:"all .15s",whiteSpace:"nowrap"});
  const smBtn={padding:"4px 10px",borderRadius:6,border:`1px solid ${C.border}`,background:"transparent",fontSize:11,color:C.muted,cursor:"pointer",fontFamily:"inherit"};

  return <div style={{display:"flex",flexDirection:"column",gap:14}}>
    <div style={{"--accent":item.color,background:C.card,border:`1.5px solid ${C.border}`,borderTop:`7px solid ${item.color}`,borderRadius:16,padding:22,boxShadow:C.shadow}}>
      <div style={{fontSize:22,fontWeight:800,color:item.color,marginBottom:2}}>
        {item.label} {item.labelParen&&<span style={{fontWeight:500,fontSize:15,color:C.sub}}>{item.labelParen}</span>}
      </div>
      <div style={{fontSize:14,color:C.sub,marginBottom:14,fontStyle:"italic"}}>{item.summary}</div>
      <Hd>What it means</Hd><p style={pa}>{item.explanation}</p>
      <div style={{marginTop:14}}><Hd>Reading the plot</Hd><div style={{...pa,color:C.sub,display:"flex",flexDirection:"column",gap:4}}>{item.plotGuide.split("\n").map((line,i)=><div key={i}><MdBold text={line}/></div>)}</div></div>
      <div style={{marginTop:14}}><Hd>Formal test</Hd><p style={pa}>{item.formalTest}.</p></div>
      <div style={{marginTop:14}}><Hd>What goes wrong when violated</Hd><p style={pa}><MdBold text={item.whatBreaks}/></p></div>
      <div style={{marginTop:14}}><Hd>How to fix</Hd><p style={pa}><MdBold text={item.howToFix}/></p></div>
      <Quiz questions={MCQ[item.key]} color={item.color}/>
      <div style={{marginTop:18,borderTop:`1px solid ${C.border}`,paddingTop:14}}>
        <div style={{fontSize:11,fontWeight:700,color:C.muted,letterSpacing:.8,marginBottom:4}}>
          EXAMPLE DATASETS <span style={{fontWeight:400,fontStyle:"italic",letterSpacing:0}}>(simulated for illustration)</span>
        </div>
        <div style={{display:"flex",gap:8,flexWrap:"wrap",marginBottom:10}}>
          {tabOrder.map(k=><button key={k} style={pill(k)} onClick={()=>loadEx(k)}><span style={{marginRight:5}}>{tabIcons[k]}</span>{tabLabels[k]}</button>)}
        </div>
        {ds&&<div style={{background:`${item.colorSoft}55`,borderRadius:10,padding:"10px 14px"}}>
          <div style={{fontSize:14,fontWeight:700,color:item.color,marginBottom:2}}>{ds.label}</div>
          <div style={{fontSize:13,color:C.sub,lineHeight:1.55}}><b>{ds.desc}</b></div>
        </div>}
        {custom&&<div style={{fontSize:13,color:C.muted,marginTop:8,fontStyle:"italic"}}>You have modified the data. Select an example above to reset.</div>}
      </div>
    </div>
    <div style={{display:"grid",gridTemplateColumns:"minmax(520px, 1.1fr) minmax(470px, 1fr)",gap:14,alignItems:"start",minWidth:1000}}>
      <div style={{background:C.card,border:`1.5px solid ${C.border}`,borderRadius:16,padding:14,boxShadow:C.shadow}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:4,flexWrap:"wrap",gap:6}}>
          <span style={{fontSize:15,fontWeight:700,color:C.text}}>Scatter Plot</span>
          <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
            <button onClick={handleDownload} disabled={!points.length} title="Download the current points as a CSV" style={smBtn}>Download CSV</button>
            <button onClick={handleCopyR} disabled={!points.length} title="Copy R code that recreates these points as a data.frame" style={smBtn}>{copied?"Copied!":"Copy R code"}</button>
            <button onClick={()=>{setPoints([]);setCustom(true);setHiIdx(null);}} style={smBtn}>Clear all</button>
          </div>
        </div>
        <div style={{display:"flex",gap:10,marginBottom:8,fontSize:12,color:C.sub,flexWrap:"wrap"}}>
          <span><b>+</b> Double-click to add</span>
          <span><b>&harr;</b> Drag to move</span>
          <span><b>&times;</b> Double-click point to remove</span>
          <span><b>&bull;</b> Click point to track across plots</span>
        </div>
        <Scatter points={points} setPoints={setPoints} model={model} hiIdx={hiIdx} onHi={setHiIdx} onEdit={onEdit}/>
        {model&&<div style={{marginTop:8,fontFamily:"'JetBrains Mono',monospace",fontSize:14,color:C.sub,padding:"6px 12px",background:C.bg,borderRadius:6}}>
          y&#770; = {model.b0.toFixed(2)} + {model.b1.toFixed(2)}x &nbsp;&nbsp;&nbsp; n = {points.length}
        </div>}
      </div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
        {["linearity","homogeneity","influential","normality"].map(dt=><DiagPlot key={dt} type={dt} model={model} highlight={dt===item.diagKey?item.color:null} hiIdx={hiIdx} onHi={setHiIdx}/>)}
      </div>
    </div>
  </div>;
}

/* ── MAIN ────────────────────────────────────────────────────────────── */
export default function App(){
  const[selected,setSelected]=useState("shape");
  const[points,setPoints]=useState([...DS.lin_good.points]);
  const model=useMemo(()=>ols(points),[points]);
  const selItem=SUNSHINE.find(s=>s.key===selected);
  const[showNote,setShowNote]=useState(false);
  const[customData,setCustomData]=useState(false);
  const handleSel=k=>{if(k===selected)return;const it=SUNSHINE.find(s=>s.key===k);if(customData&&it?.type==="diagnostic"&&!window.confirm("Switching assumptions will replace your modified points with that section's example data. Continue?"))return;setSelected(k);if(it?.type==="diagnostic"&&it.examples?.good){const d=DS[it.examples.good];if(d){setPoints([...d.points]);setCustomData(false);}}};

  return <div style={{fontFamily:"Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",background:C.bg,minHeight:"100vh",minWidth:1120,color:C.text}}>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
    <div style={{height:5,background:`linear-gradient(90deg, ${C.gold}, ${C.tealLight}, ${C.teal})`}}/>
    <div style={{width:1060,minWidth:1060,margin:"0 auto",padding:"20px 28px 40px"}}>
      <div style={{marginBottom:20}}>
        <h1 style={{fontSize:28,fontWeight:800,margin:0,lineHeight:1.15,color:C.teal,letterSpacing:"-0.02em"}}>Regression Assumptions Explorer</h1>
        <p style={{fontSize:14,color:C.sub,margin:"6px 0 0",lineHeight:1.5}}>Click any letter below to explore that assumption. Simple linear regression focus.</p>
      </div>
      <div style={{display:"flex",gap:5,marginBottom:20,flexWrap:"wrap"}}>
        {SUNSHINE.map(item=>{const isA=selected===item.key;const isD=item.type==="diagnostic";return <button key={item.key} onClick={()=>handleSel(item.key)} style={{display:"flex",flexDirection:"column",alignItems:"center",padding:"10px 6px 6px",borderRadius:12,minWidth:56,flex:"1 1 0",border:`2.5px solid ${isA?item.color:C.border}`,background:isA?item.colorSoft:C.card,cursor:"pointer",transition:"all .2s",position:"relative",boxShadow:isA?`0 3px 16px ${item.color}25`:"0 1px 3px #0001",transform:isA?"translateY(-2px)":"none"}}>
          <span style={{fontWeight:800,fontSize:27,lineHeight:1,color:isA?item.color:"#A09A90"}}>{item.letter}</span>
          <span style={{fontSize:8.5,color:isA?item.color:C.muted,textAlign:"center",lineHeight:1.2,marginTop:3,fontWeight:isA?800:600,maxWidth:76}}>{item.label}</span>
          {isD&&<MiniIcon type={item.diagKey}/>}
          {isA&&<div style={{position:"absolute",bottom:-10,left:"50%",transform:"translateX(-50%)",width:0,height:0,borderLeft:"9px solid transparent",borderRight:"9px solid transparent",borderTop:`9px solid ${item.color}`}}/>}
        </button>;})}
      </div>
      {selItem?.type==="info"&&<InfoPanel item={selItem}/>}
      {selItem?.type==="diagnostic"&&<DiagPanel key={selItem.key} item={selItem} points={points} setPoints={setPoints} model={model} custom={customData} setCustom={setCustomData}/>}
      {selItem?.type==="diagnostic"&&<div style={{marginTop:22}}>
        <button onClick={()=>setShowNote(!showNote)} style={{padding:"8px 16px",borderRadius:20,border:`1.5px solid ${C.border}`,background:C.card,fontSize:13,color:C.sub,cursor:"pointer",fontFamily:"inherit"}}>
          {showNote?"Hide note":"A note on reading diagnostic plots"}
        </button>
        {showNote&&<div style={{marginTop:8,padding:"14px 18px",background:C.card,border:`1.5px solid ${C.border}`,borderRadius:12,fontSize:14,color:C.sub,lineHeight:1.7,maxWidth:700}}>
          <p style={{margin:"0 0 8px"}}>Reading diagnostic plots is more of a learned skill than a mechanical pass/fail test. Two analysts can look at the same plot and disagree about whether a violation is present. With practice you develop a sense for what random scatter looks like versus a real pattern.</p>
          <p style={{margin:0}}>If linear regression is central to your analysis, it is good practice to <b>report your assumption checks</b> (typically in supplementary materials) so readers can judge for themselves.</p>
        </div>}
      </div>}
    </div>
  </div>;
}

const rootEl = document.getElementById("root");
if (rootEl) createRoot(rootEl).render(<App />);
