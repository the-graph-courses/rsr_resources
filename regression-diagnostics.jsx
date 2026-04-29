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
lin_good:{label:"Exercise & Cardiovascular Fitness",desc:"Weekly exercise hours vs. VO\u2082 max (mL/kg/min) for 30 adults in a wellness program. A clean, linear positive relationship.",xLabel:"Weekly exercise hours",yLabel:"VO\u2082 max (mL/kg/min)",points:(()=>{const xs=[1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,2,3,4.5,6,7,8,9,1.5,5,8.5];const ns=[2,-1,3,-2,1,-3,2,-1,4,-2,1,-3,2,-4,3,-1,2,-2,1,-3,0,2,-1,3,-2,1,-1,2,-3,1];return xs.map((x,i)=>({x,y:+(28+1.6*x+ns[i]*1.0).toFixed(1)}));})()},
lin_border:{label:"Age & Systolic Blood Pressure",desc:"Age vs. systolic BP from a community screening. BP rises with age, possibly accelerating slightly. Subtle enough to debate.",xLabel:"Age (years)",yLabel:"Systolic BP",points:(()=>{const xs=[25,28,31,34,37,40,43,46,49,52,55,58,61,64,67,70,27,33,39,45,51,57,63,69,30,36,42,48,54,60];const ns=[3,-2,4,-3,1,-4,2,-1,5,-3,2,-5,3,-1,4,-2,5,-3,2,-4,3,-2,1,-3,4,-2,2,-3,1,-4];return xs.map((x,i)=>({x:+x.toFixed(0),y:+(100+.45*x+.003*x*x+ns[i]*2.5).toFixed(0)}));})()},
lin_bad:{label:"Study Hours & Quiz Score",desc:"Weekly study hours vs. quiz score (%). Going from almost no studying to moderate studying lifts scores sharply, while extra hours past that add little, since you are brushing the top of the scale. That bend is nowhere near straight.",xLabel:"Weekly study hours",yLabel:"Quiz score (%)",points:(()=>{const xs=[2,4,6,9,11,13,15,17,18,20,21,23,26,29,31,34,37,39,41,43,46,48,51,53,54,56,58,60,61,62];const ns=[.8,-1,1,-.8,.5,-1,.6,-1.2,.4,-1,.7,-1.1,.9,-1,.5,-1.2,.8,-.9,.6,-1,.5,-1.1,.9,-1.2,.8,-1,.6,-1.1,.9,-1.2];return xs.map((x,i)=>{const y=96*(1-Math.exp(-x/9.8))+ns[i]*2.2;return{x,y:+Math.min(y,99.9).toFixed(1)};});})()},
hom_good:{label:"Exercise Duration & Calories Burned",desc:"Minutes of exercise vs. calories burned. The spread stays consistent from short to long workouts.",xLabel:"Exercise duration (minutes)",yLabel:"Calories burned",points:(()=>{const xs=[15,20,22,25,28,30,33,35,37,40,42,45,48,50,52,55,58,60,63,65,18,27,34,39,44,49,54,57,62,36];const ns=[12,-8,15,-10,6,-14,9,-7,16,-11,5,-13,10,-6,14,-9,7,-12,11,-8,13,-10,8,-15,6,-11,14,-7,9,-12];return xs.map((x,i)=>({x,y:+(50+5.8*x+ns[i]).toFixed(0)}));})()},
hom_border:{label:"Education & Health Literacy",desc:"Years of education vs. health literacy score. Scores widen slightly at higher education levels, but the pattern is subtle.",xLabel:"Years of education",yLabel:"Health literacy score",points:(()=>{const xs=[8,9,10,10,11,11,12,12,12,13,13,14,14,14,15,15,16,16,16,17,17,18,18,19,20,9,11,13,15,17];const ns=[1.5,-1,2,-1.5,2.5,-2,1.5,-2.5,3,-1,2.5,-2,3,-3,2,-3.5,3.5,-2.5,4,-3,3.5,-4,3,-4.5,5,1,-2,2.5,-3,3.5];return xs.map((x,i)=>({x,y:+(20+4*x+ns[i]*(1+(x-14)*.04)).toFixed(1)}));})()},
hom_bad:{label:"Income & Medical Spending",desc:"Household income ($K) vs. annual out-of-pocket medical spending. Wealthier households show much more variable spending, creating a clear fan shape.",xLabel:"Household income ($K)",yLabel:"Medical spending ($)",points:(()=>{const xs=[20,25,28,32,35,38,42,45,48,52,55,58,62,65,68,72,75,80,85,90,95,100,110,120,22,40,56,70,88,105];const ns=[-.6,1.1,-.4,1.3,-.9,.7,-1.1,1.5,-.7,1.2,-1.5,.9,1.7,-1.3,-2,2.3,-1.2,2.6,-1.7,-2.7,2.1,-3,3.6,1.8,-.7,-1.5,2.4,-2.1,2.9,-3.5];const ms=[7,-11,9,-4,-13,15,-7,17,3,-15,-9,13,-17,11,5,-3,-19,9,-7,13,-11,17,-15,11,19,-5,13,-17,15,-9];return xs.map((x,i)=>({x,y:+(200+12*x+ns[i]*x*1.5+ms[i]*6).toFixed(0)}));})()},
inf_good:{label:"BMI & Total Cholesterol",desc:"BMI vs. total cholesterol (mg/dL) from a routine health screening. No point exerts outsized influence.",xLabel:"BMI",yLabel:"Total cholesterol (mg/dL)",points:(()=>{const xs=[19,20,21,21.5,22,22.5,23,23.5,24,24.5,25,25.5,26,26.5,27,27.5,28,28.5,29,29.5,30,30.5,31,31.5,32,20.5,23.5,26,28,30];const ns=[5,-3,7,-4,2,-6,4,-2,8,-5,3,-7,5,-3,6,-4,7,-5,3,-6,8,-4,5,-7,6,4,-5,3,-4,7];return xs.map((x,i)=>({x,y:+(120+4.5*x+ns[i]*2).toFixed(0)}));})()},
inf_border:{label:"Unusually Healthy Elder",desc:"Age vs. systolic BP, with one 78-year-old whose BP is lower than expected. High leverage, but roughly on trend.",xLabel:"Age (years)",yLabel:"Systolic BP",points:(()=>{const xs=[30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,33,39,45,51,57,63,35,43,55,61];const ns=[3,-2,4,-3,1,-4,2,-1,5,-3,2,-4,3,-2,4,-1,5,-3,2,-4,3,-2,1,-3,4,-2,2,-3,1,-4];const p=xs.map((x,i)=>({x,y:+(90+.8*x+ns[i]*2).toFixed(0)}));p.push({x:78,y:140});return p;})()},
inf_bad:{label:"Data Entry Error in BMI Study",desc:"Same cholesterol screening, but one record has BMI=42 with cholesterol=130. Likely a data entry error. Cook's distance flags it immediately.",xLabel:"BMI",yLabel:"Total cholesterol (mg/dL)",points:(()=>{const xs=[19,20,21,22,23,23.5,24,24.5,25,25.5,26,26.5,27,27.5,28,28.5,29,29.5,30,30.5,31,20.5,23,25.5,27,29,22,24.5,26.5,28.5];const ns=[5,-3,7,-4,2,-6,4,-2,8,-5,3,-7,5,-3,6,-4,7,-5,3,-6,8,4,-5,3,-4,7,-3,6,-5,4];const p=xs.map((x,i)=>({x,y:+(120+4.5*x+ns[i]*2).toFixed(0)}));p.push({x:42,y:130});return p;})()},
norm_good:{label:"Height & Lung Capacity",desc:"Height (cm) vs. lung capacity (L) for 60 adults. Prediction errors distribute symmetrically.",xLabel:"Height (cm)",yLabel:"Lung capacity (L)",points:(()=>{let s=20260429;const r=()=>{s=(s*1664525+1013904223)>>>0;return s/4294967296;};const z=()=>(Math.sqrt(-2*Math.log(Math.max(r(),1e-9)))*Math.cos(2*Math.PI*r())+Math.sqrt(-2*Math.log(Math.max(r(),1e-9)))*Math.cos(2*Math.PI*r()))/Math.SQRT2;const n=60;const xs=Array.from({length:n},(_,i)=>155+(i/(n-1))*35+(r()-.5)*1.2);const raw=xs.map(()=>z());const m=raw.reduce((a,v)=>a+v,0)/n;const sd=Math.sqrt(raw.reduce((a,v)=>a+(v-m)**2,0)/(n-1))||1;return xs.map((x,i)=>{const std=(raw[i]-m)/sd;return{x:+x.toFixed(1),y:+(-3+.042*x+std*0.18).toFixed(2)};});})()},
norm_border:{label:"Hours of Sleep & Sick Days",desc:"Average nightly sleep vs. sick days per year. Most people follow the trend, with only a mild tail wiggle.",xLabel:"Average nightly sleep (hours)",yLabel:"Sick days per year",points:(()=>{const xs=[5,5.5,5.5,6,6,6.5,6.5,6.5,7,7,7,7,7.5,7.5,7.5,7.5,8,8,8,8,8.5,8.5,8.5,9,9,5,6,7,8,8.5];const ns=[-1.1,.2,-.3,.45,-.1,.3,-.2,.5,-.15,.1,-.3,.25,-.4,.15,.05,-.25,.35,-.1,.2,-.35,1.0,.3,-.2,-.15,-.4,.9,.4,-.05,.15,-.8];return xs.map((x,i)=>({x,y:+(18-1.5*x+ns[i]*1.1).toFixed(1)}));})()},
norm_bad:{label:"Poverty Rate & ER Visits",desc:"Neighborhood poverty rate (%) vs. ER visits per 1,000. Most follow the trend, but several neighborhoods have extremely high visit rates. Clearly right-skewed residuals.",xLabel:"Neighborhood poverty rate (%)",yLabel:"ER visits per 1,000",points:(()=>{let s=20260512;const r=()=>{s=(s*1664525+1013904223)>>>0;return s/4294967296;};const z=()=>Math.sqrt(-2*Math.log(Math.max(r(),1e-9)))*Math.cos(2*Math.PI*r());const xs=[5,7,8,10,12,13,15,16,18,19,20,22,23,25,27,28,30,32,34,35,37,38,40,42,6,14,21,29,36,41];const skewIdx=new Set([4,8,12,16,20,26,29]);return xs.map((x,i)=>{const noise=z()*9;const skew=skewIdx.has(i)?60+r()*55:0;return{x,y:Math.max(0,Math.round(80+3.5*x+noise+skew))};});})()},
};

/* ── MCQ DATA ────────────────────────────────────────────────────────── */
const MCQ={
sample:[
{q:"Both panels show the same true relationship; a single new red point has been added to each. In which panel did the new point most reshape the fitted line?",v:"sample",opts:["The n = 8 panel","The n = 80 panel"],ans:0,explain:"With only 8 observations, one unusual point can noticeably rotate the fitted line. With 80 observations, the same kind of point has much less leverage over the whole estimate."},
{q:"The two intervals shown are centred on the same slope estimate. Which sample size produced the wider confidence interval?",v:"sample_ci",opts:["n = 200","n = 15"],ans:1,explain:"The point estimate is the same, but the smaller sample has more uncertainty, so its confidence interval is wider."},
{q:"Which statement about sample size is most accurate?",opts:["You should almost never fit regression with fewer than 30 observations","Small samples can be fit, but estimates and diagnostics are fragile","Sample size only matters in multiple regression"],ans:1,explain:"There is no magic cutoff. The key teaching point is that small samples make slopes, standard errors, and diagnostic plots less stable."},
],
uncorrelated:[
{q:"You collect patient outcomes from 10 clinics and fit a single OLS regression that ignores which clinic each patient came from. Patients from the same clinic share staff, scheduling, and case mix. Which assumption is most directly threatened?",opts:["Independence of errors","Linearity of the relationship","Normality of residuals"],ans:0,explain:"Patients from the same clinic share context, so their residuals tend to look alike. That makes the rows non-independent, which inflates apparent precision (standard errors are too small). Linearity and normality are not the main issue here."},
{q:"Residuals plotted in collection order move in runs of similar values rather than flipping sign at each step. What does this suggest?",v:"time",opts:["Heteroscedasticity: the spread of residuals changes along the sequence","Nearby rows have residuals on the same side of zero, so they are not independent","Omitted variable bias: a missing predictor that increases with row order"],ans:1,explain:"Runs and waves in collection order mean neighbouring errors are positively related (autocorrelation), not independent. Heteroscedasticity is a changing spread, which is not the main feature here. A predictor that only trends upward would not produce alternating runs above and below zero."},
],
multicollinearity:[
{q:"A researcher fits the model:\u00A0\u00A0blood_pressure = \u03B2\u2080 + \u03B2\u2081\u00B7age + \u03B5. Why can multicollinearity not occur in this model?",opts:["There is only one predictor","The predictor is normally distributed"],ans:0,explain:"Multicollinearity means predictors overlap with other predictors. A simple regression has only one predictor, so there is no second predictor for it to be collinear with."},
{q:"A model is fit as:\u00A0\u00A0health_score = \u03B2\u2080 + \u03B2\u2081\u00B7education + \u03B2\u2082\u00B7income + \u03B5. Education and income turn out to be very strongly correlated with each other in the data. What does this imply for the coefficient estimates?",opts:["The two predictors carry overlapping information, making it hard to estimate their separate effects","The predictors will jointly produce non-normal residuals"],ans:0,explain:"The overlap means the predictors move together, so the model has trouble deciding which predictor deserves credit for the shared part of the signal. The coefficient standard errors inflate."},
{q:"Education and income are strongly correlated in your data and you include both as predictors. What should you expect compared with a model that uses only one of them?",opts:["Larger standard errors on each coefficient, with little change in R\u00B2","Roughly double the explanatory power, since each predictor adds new information"],ans:0,explain:"Highly correlated predictors leave the model unable to separate their individual contributions, which inflates the standard errors. The shared information is largely already in either predictor alone, so R\u00B2 often does not increase by much."},
],
shape:[
{q:"The dots clearly trace a curve, but the OLS line is straight. What does that say about the slope?",v:"curve",opts:["It misrepresents the relationship, because the true slope changes across x. A common fix is to let the model bend, for example by log-transforming x or adding a polynomial term","It is still valid as an average rate of change, since OLS gives the best straight-line fit and the points scatter around it"],ans:0,explain:"A straight line forces one average slope onto a relationship whose slope actually changes across x. Common fixes: log or square-root transform of the predictor, or a polynomial term such as x squared."},
{q:"Looking at this residuals-vs-fitted plot, what reassures you that linearity holds?",v:"resid_good",opts:["The smoother stays close to zero across all fitted values","The positive and negative residuals balance out, summing to roughly zero"],ans:0,explain:"A flat smoother near zero means no systematic pattern is left in the residuals. Residuals summing to zero is automatic for any OLS fit, so it says nothing about linearity. A clear U-shape sums to zero just as easily."},
],
homogeneity:[
{q:"Three residual-vs-fitted patterns are shown. Which one most clearly indicates heteroscedasticity?",v:"het_pick",opts:["The middle panel","The left panel","The right panel"],ans:0,explain:"Heteroscedasticity means the spread of residuals changes with the fitted value. The middle panel shows residuals widening from left to right, the classic fan. The left panel has roughly constant spread (homoscedastic), and the right panel shows a U-shape, which is a linearity issue rather than a variance one."},
{q:"When residual variance is clearly non-constant, why are the usual confidence intervals unreliable?",opts:["The standard formula assumes constant variance, so the resulting interval can be too narrow or too wide","The slope estimate itself becomes biased, so any interval centred on it lands in the wrong place"],ans:0,explain:"OLS standard errors are derived under the assumption of constant variance, and heteroscedasticity distorts that calculation. Robust (HC) standard errors recompute it without that assumption. The slope itself can still be unbiased."},
{q:"Why might income vs. medical spending look like a fan-shaped scatter?",v:"income_fan",opts:["Higher-income households have more discretionary spending, so individual choices add more variability at the top","Higher-income households spend more on average, so the regression line is steeper at the high end"],ans:0,explain:"The fan is about variance, not the mean. At higher incomes, spending depends more on personal choice, which inflates the spread. A steeper average slope at the top would be a linearity issue, not heteroscedasticity."},
],
influential:[
{q:"Three panels show different points marked in red. Which point is most influential on the fitted line?",v:"outlier_types",opts:["The low-leverage outlier (left panel)","The high-leverage point on the line (middle panel)","The off-trend point with high leverage (right panel)"],ans:2,explain:"Influence usually requires both ingredients: unusual x-position and a sizable residual. The right-panel point has both."},
{q:"Which pair of ingredients does Cook's distance combine to flag a point as influential?",ev:"cooks_combine",opts:["The number of predictors and the residual standard error","Leverage (unusual X) and residual size (off the line)","The sample size and the model's R\u00B2"],ans:1,explain:"Cook's distance combines how unusual a point's X value is (leverage) with how far its Y sits from the fitted line (residual). A point needs both to meaningfully move the line: high leverage with a tiny residual barely shifts the fit, and a big residual at an average X value pulls less than one at the edge."},
{q:"In this leverage-vs-residual plot, the red point sits near the Cook's distance contour. Why is that a concern?",v:"lev_resid",opts:["Its X value is unusual and its residual is large, so removing it could meaningfully shift the fitted line","Its leverage alone is enough to flag it, regardless of where it sits relative to the line"],ans:0,explain:"The Cook's distance contour combines leverage and residual size. A point sitting near it has unusual amounts of both, so dropping it could meaningfully change the slope. Leverage on its own is not enough."},
{q:"The figure below shows the same data fitted with and without the red point. What is the danger this illustrates for the conclusions you would report?",v:"slope_shift",opts:["The reported slope may rest heavily on a single observation rather than on the overall pattern","Removing flagged points always tightens the confidence interval and improves R\u00B2","The point should be deleted before reporting any results"],ans:0,explain:"The concern is that one observation can change the headline conclusion. R\u00B2 and CI behaviour after removal depends on the data, and silently dropping flagged points is not good practice. The right response is usually to investigate the point and report sensitivity of the result with and without it."},
],
normality:[
{q:"In this detrended Q-Q plot, the points trail away from the zero line at both ends, while the middle stays close to it. What is this most likely a sign of?",v:"qq_heavy_tails",opts:["Heavier tails than a normal distribution","Approximately normal residuals"],ans:0,explain:"In a detrended Q-Q plot, the diagonal of the regular Q-Q has been flattened to a horizontal zero line. Points pulling away at both ends are the signature of heavier-than-normal tails (often extreme outliers). The middle staying close to zero rules out a strong overall skew."},
{q:"In this detrended Q-Q plot, the points scatter randomly around the horizontal zero line, mostly within the shaded confidence band. What does that suggest about the residuals?",v:"worm",opts:["Roughly normal residuals","Heavy-tailed residuals","Skewed residuals"],ans:0,explain:"In the detrended version, the diagonal of a regular Q-Q has been flattened to horizontal. Random bouncing around zero with most points inside the confidence band is the signature of approximate normality. Heavy tails would push the ends away from zero, and a skewed distribution would tilt the points systematically above or below zero across the plot."},
{q:"For which sample size does the normality assumption matter most?",ev:"clt_safety",opts:["The small (n = 10) sample","The large (n = 200) sample, because of the CLT"],ans:0,explain:"With small samples, p-values and confidence intervals lean more heavily on the normality assumption. With large samples, the CLT often helps, but it is not a magic fix for extreme outliers, dependence, or the wrong model shape."},
{q:"What does non-normality of residuals primarily affect?",opts:["The reported confidence intervals and p-values","The fitted predictions, but not the standard errors"],ans:0,explain:"The fitted line can still be useful for prediction. The bigger concern is whether the usual uncertainty statements are trustworthy."},
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
{key:"sample",letter:"S",label:"Sample Size",type:"info",color:"#9B6B2F",colorSoft:"#F5E6D0",summary:"Do you have enough observations for your model?"},
{key:"uncorrelated",letter:"U",label:"Uncorrelated Errors",type:"info",color:"#2E86AB",colorSoft:"#D0EAF5",summary:"Are residuals independent of each other?"},
{key:"multicollinearity",letter:"N",label:"No Multicollinearity",type:"info",color:"#6B6B6B",colorSoft:"#ECECEC",summary:"Are predictors too correlated? (Multiple regression only)"},
{key:"shape",letter:"S",label:"Linearity",labelParen:"(Shape)",type:"diagnostic",diagKey:"linearity",color:"#2B6CB0",colorSoft:"#D4E3F5",summary:"Is the relationship actually a straight line?",
  readingPlotNames:["Residuals vs fitted","Residual plot"],
  explanation:"The relationship between predictor and outcome should follow a straight line. With one predictor you can sometimes spot curvature in the scatter plot, but the fitted-vs-residuals plot shows it more clearly: a smooth line near zero means linear, a U-shape or arch means curved.",
  plotGuide:"**X-axis: fitted values.** The model's predicted y for each observation.\n**Y-axis: residuals.** Actual minus predicted y, the part of the data the line did not capture.\nThe dashed horizontal line at zero marks perfect predictions. The green smooth line traces the average residual at each fitted value. If it curves, the real relationship is not linear.",
  whatBreaks:{items:[
    {status:"bad",text:"**Slope:** biased, since a single line cannot capture a curved relationship."},
    {status:"bad",text:"**SEs, CIs, and p-values:** unreliable, since they are built on the wrong model."}
  ],bottomLine:"predictions are systematically off in some regions of x."},
  formalTestList:[
    {text:"**Ramsey RESET test**.",
      links:[{title:"lmtest::resettest reference (CRAN)",url:"https://search.r-project.org/CRAN/refmans/lmtest/html/resettest.html"},{title:"MetricGate: RESET test tutorial in R",url:"https://metricgate.com/docs/reset-test/"}]}
  ],
  howToFixList:[
    {text:"**Transform the predictor** (log, square root, reciprocal) to linearise the relationship.",
      links:[{title:"UVA Library: Interpreting log transformations",short:"UVA logs",url:"https://library.virginia.edu/data/articles/interpreting-log-transformations-in-a-linear-model"},{title:"Penn State STAT 462: Log-transforming the predictor",short:"PSU",url:"https://online.stat.psu.edu/stat462/node/152/"}]},
    {text:"**Add a polynomial term** (for example a quadratic in the predictor).",
      links:[{title:"Statology: Polynomial regression in R",short:"Statology",url:"https://www.statology.org/polynomial-regression-r/"},{title:"STHDA: Polynomial and spline regression in R",short:"STHDA",url:"https://www.sthda.com/english/articles/40-regression-analysis/162-nonlinear-regression-essentials-in-r-polynomial-and-spline-regression-models/"}]},
    {text:"**Fit a non-linear model** when a straight line or low-order polynomial is not enough.",
      links:[{title:"DataScience+: First steps with non-linear regression in R",short:"DataScience+",url:"https://datascienceplus.com/first-steps-with-non-linear-regression-in-r/"},{title:"stats::nls reference (rdrr)",short:"nls()",url:"https://rdrr.io/r/stats/nls.html"}]}
  ],
  examples:{good:"lin_good",borderline:"lin_border",bad:"lin_bad"}},
{key:"homogeneity",letter:"H",label:"Homogeneity of Variance",labelParen:"(Homoscedasticity)",type:"diagnostic",diagKey:"homogeneity",color:"#2F855A",colorSoft:"#D4EDDF",summary:"Does the spread of residuals stay constant?",
  readingPlotNames:["Scale–location plot","\u221A|standardized residuals| vs fitted"],
  explanation:"The spread of prediction errors should stay roughly constant across the range of the predictor. If the residuals fan out or shrink (heteroscedasticity), the standard errors are wrong, so confidence intervals and p-values cannot be trusted.",
  plotGuide:"**X-axis: fitted values.** The model's predicted y for each observation.\n**Y-axis: \u221A|standardized residuals|.** Error magnitude on a z-score-like scale, with the square root compressing the tail so your eye can judge whether the smooth line is flat.\nA flat line means constant spread. An upward slope means variance is growing with the fitted value.",
  whatBreaks:{items:[
    {status:"good",text:"**Slope:** typically still unbiased."},
    {status:"bad",text:"**SEs, CIs, and p-values:** the SE formula is wrong (often understated), so CIs are too narrow and p-values too small."}
  ],bottomLine:"the line itself is roughly right, but the uncertainty around it cannot be trusted."},
  formalTestList:[
    {text:"**Breusch-Pagan test**.",
      links:[{title:"Statology: Breusch-Pagan test in R",url:"https://www.statology.org/breusch-pagan-test-r/"},{title:"lmtest::bptest reference (CRAN)",url:"https://search.r-project.org/CRAN/refmans/lmtest/html/bptest.html"}]}
  ],
  howToFixList:[
    {text:"Use **heteroscedasticity-robust standard errors** (HC variants such as HC3).",
      links:[{title:"UVA Library: Understanding robust standard errors",short:"UVA robust SE",url:"https://library.virginia.edu/data/articles/understanding-robust-standard-errors"},{title:"r-econometrics: HC robust errors with sandwich",short:"HC errors",url:"https://www.r-econometrics.com/methods/hcrobusterrors/"}]},
    {text:"**Transform the outcome** (log often stabilises variance).",
      links:[{title:"EPsy 8252: Log-transforming the outcome",short:"EPsy 8252",url:"https://zief0002.github.io/book-8252/nonlinearity-log-transforming-the-outcome.html"},{title:"UVA Library: Interpreting log transformations",short:"UVA logs",url:"https://library.virginia.edu/data/articles/interpreting-log-transformations-in-a-linear-model"}]},
    {text:"Fit **weighted least squares** when you can model how variance depends on the predictors or fitted values.",
      links:[{title:"Statology: Weighted least squares in R",short:"Statology",url:"https://www.statology.org/weighted-least-squares-in-r/"},{title:"R-bloggers: Conquering unequal variance with WLS in R",short:"R-bloggers",url:"https://www.r-bloggers.com/2023/12/conquering-unequal-variance-with-weighted-least-squares-in-r-a-practical-guide/"}]}
  ],
  examples:{good:"hom_good",borderline:"hom_border",bad:"hom_bad"}},
{key:"influential",letter:"I",label:"Influential Points",type:"diagnostic",diagKey:"influential",color:"#C53030",colorSoft:"#FED7D7",summary:"Is any single point pulling the regression line too much?",
  readingPlotNames:["Residuals vs leverage","Leverage plot"],
  explanation:"An influential observation is one whose removal would noticeably change the fitted line. The biggest culprits combine two features: an unusual X value (high leverage) and a sizable residual (off where the line would otherwise sit). High leverage on its own does not always make a point influential, but when paired with a large residual it almost always does.",
  plotGuide:"**X-axis: leverage.** How unusual the observation's predictor value is.\n**Y-axis: standardized residual.** How far the observation sits from the fitted line, on a z-score-like scale.\nThe dashed curve is a Cook's distance contour at the F(0.5, p, n−p) quantile (matching R's performance::check_model). Points beyond the curve combine high leverage with a large residual, the two ingredients of an influential observation.",
  whatBreaks:{items:[
    {status:"bad",text:"**Slope:** can shift substantially when a single influential point is removed."},
    {status:"bad",text:"**SEs, CIs, and p-values:** also unstable, since they too depend on that one point."}
  ],bottomLine:"your headline conclusion may rest on a single observation."},
  formalTestList:[
    {text:"**Cook's distance**: overall influence on fitted values. Typical threshold for simple regression: F(0.5, p, n−p) ≈ 0.7.",
      links:[{title:"Statology: Identify influential points with Cook's distance",url:"https://www.statology.org/how-to-identify-influential-data-points-using-cooks-distance/"},{title:"RMPH §5.23: Influential observations",url:"https://bookdown.org/rwnahhas/RMPH/mlr-influence.html"}]},
    {text:"**DFFITS**: change in the fitted value when point i is dropped.",
      links:[{title:"olsrr vignette: Measures of influence in R",url:"https://cran.r-project.org/web/packages/olsrr/vignettes/influence_measures.html"},{title:"stats::influence.measures (R manual)",url:"https://stat.ethz.ch/R-manual/R-devel/library/stats/html/influence.measures.html"}]},
    {text:"**DFBETAS**: change in each coefficient when point i is dropped.",
      links:[{title:"UVA Library: Detecting influential points with DFBETAS",url:"https://library.virginia.edu/data/articles/detecting-influential-points-in-regression-with-dfbetas"},{title:"Statology: How to calculate DFBETAS in R",url:"https://www.statology.org/dfbetas-in-r/"}]}
  ],
  howToFixList:[
    {text:"**Investigate flagged points** to understand why they are unusual.",
      links:[{title:"Stats and R: Outliers detection in R",short:"Stats and R",url:"https://statsandr.com/blog/outliers-detection-in-r/"},{title:"RMPH §5.22: Outliers (bookdown)",short:"RMPH",url:"https://bookdown.org/rwnahhas/RMPH/mlr-outliers.html"}]},
    {text:"If they look like **data errors, correct or remove them** and refit.",
      links:[{title:"r-statistics.co: Outlier detection and treatment in R",short:"r-statistics.co",url:"https://r-statistics.co/Outlier-Treatment-With-R.html"},{title:"UCLA OARC: Robust regression in R",short:"UCLA",url:"https://stats.oarc.ucla.edu/r/dae/robust-regression/"}]},
    {text:"If they look real, **report results with and without** them (sensitivity analysis).",
      links:[{title:"RMPH §5.26: Sensitivity analysis",short:"RMPH",url:"https://www.bookdown.org/rwnahhas/RMPH/mlr-sensitivity.html"}]}
  ],
  examples:{good:"inf_good",borderline:"inf_border",bad:"inf_bad"}},
{key:"normality",letter:"N",label:"Normality of Residuals",type:"diagnostic",diagKey:"normality",color:"#6B46C1",colorSoft:"#E9D8FD",summary:"Do prediction errors follow a bell curve?",
  plotCaption:"Detrended normal Q–Q plot",
  readingPlotNames:["Detrended normal Q–Q plot","Normal Q–Q plot"],
  explanation:"The residuals should be roughly bell-shaped. The fitted line itself does not need perfectly normal residuals, but the usual confidence intervals and p-values do, especially in small samples. Strong skew, heavy tails, or extreme outliers make those uncertainty statements unreliable.",
  plotGuide:"**X-axis: theoretical normal quantiles.** Where each residual would sit if the residuals were exactly normal.\n**Y-axis: deviation from the normal line.** How far the observed residual quantile lands from that theoretical position. The diagonal of a regular Q-Q plot has been 'flattened' to a horizontal zero line (this is the detrended Q-Q used by performance::check_model).\nA shaded confidence band gives a tolerance. As long as most points scatter inside the band, mild departures are not a concern. A systematic curve, an S-shape, or many points outside the band suggests the residual distribution is not normal.",
  whatBreaks:{items:[
    {status:"good",text:"**Slope:** still unbiased."},
    {status:"warn",text:"**SEs, CIs, and p-values:** the SE itself is roughly right, but CIs and p-values rely on a t-distribution that only holds under normality, so they can be off in small samples."}
  ],bottomLine:"with large samples, mild deviations are usually fine because of the CLT."},
  formalTestList:[
    {text:"**Shapiro-Wilk test**.",
      links:[{title:"STHDA: Normality test in R (Shapiro-Wilk, Q-Q)",url:"https://www.sthda.com/english/wiki/normality-test-in-r"},{title:"UW-Madison SSCC: Normality diagnostics in R",url:"https://sscc.wisc.edu/sscc/pubs/RegDiag-R/normality.html"}]}
  ],
  howToFixList:[
    {text:"**Transform the outcome** (log, square root, or Box–Cox).",
      links:[{title:"Statology: Box-Cox transformation in R",short:"Statology",url:"https://www.statology.org/box-cox-transformation-in-r/"},{title:"Applied Statistics with R, Ch. 14: Transformations",short:"Stat420",url:"https://book.stat420.org/transformations.html"}]},
    {text:"**Investigate extreme values** that may be skewing the residuals.",
      links:[{title:"Stats and R: Outliers detection in R",short:"Stats and R",url:"https://statsandr.com/blog/outliers-detection-in-r/"},{title:"car::outlierTest reference",short:"outlierTest",url:"https://rdrr.io/cran/car/man/outlierTest.html"}]},
    {text:"With **large samples**, mild non-normality is often less serious. The CLT helps the sampling distribution of the coefficients.",
      links:[{title:"RMPH: Linear regression assumptions",short:"RMPH",url:"https://bookdown.org/rwnahhas/RMPH/slr-assumptions.html"}]}
  ],
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
function RLinks({links}){if(!links||!links.length)return null;const aStyle={color:"#006D77",textDecoration:"none",borderBottom:"1px dotted #83C5BE",fontSize:"0.92em"};const dot=<span style={{color:"#a8a29e",margin:"0 5px"}}>·</span>;return <span style={{fontSize:12.5,color:"#57534e",whiteSpace:"nowrap"}}>{` (e.g. `}{links.map((l,i)=><span key={i}>{i>0&&dot}<a href={l.url} target="_blank" rel="noopener noreferrer" style={aStyle} title={l.title}>{l.short||l.title}</a></span>)}{`)`}</span>;}
function FixList({items}){if(!items||!items.length)return null;return <ul style={{margin:0,paddingLeft:20,display:"flex",flexDirection:"column",gap:6}}>
  {items.map((it,i)=><li key={i} style={{fontSize:14,lineHeight:1.5,color:"#1c1917"}}>
    <span><MdBold text={it.text}/></span>{it.links&&it.links.length>0&&<RLinks links={it.links}/>}
  </li>)}
</ul>;}

function WhatBreaks({data}){
  if(!data)return null;
  if(Array.isArray(data))return <ul style={{margin:0,paddingLeft:20,fontSize:14,lineHeight:1.75,color:"#1c1917"}}>{data.map((line,i)=><li key={i}><span style={{whiteSpace:"pre"}}><MdBold text={line}/></span></li>)}</ul>;
  if(typeof data==="string")return <p style={{margin:0,fontSize:14,lineHeight:1.75,color:"#1c1917"}}><MdBold text={data}/></p>;
  const STATUS={good:{glyph:"✓",color:"#2F855A"},bad:{glyph:"✗",color:"#C53030"},warn:{glyph:"⚠",color:"#B7791F"}};
  return <div>
    <ul style={{margin:0,padding:0,listStyle:"none",display:"flex",flexDirection:"column",gap:6}}>
      {(data.items||[]).map((it,i)=>{const s=STATUS[it.status]||STATUS.bad;return <li key={i} style={{fontSize:14,lineHeight:1.5,color:"#1c1917",display:"flex",gap:8,alignItems:"flex-start"}}>
        <span aria-hidden="true" style={{color:s.color,fontWeight:700,flexShrink:0,minWidth:14,textAlign:"center",lineHeight:1.5}}>{s.glyph}</span>
        <span><MdBold text={it.text}/></span>
      </li>;})}
    </ul>
    {data.bottomLine&&<p style={{margin:"10px 0 0",fontSize:14,lineHeight:1.6,color:C.sub}}>Overall, <MdBold text={data.bottomLine}/></p>}
  </div>;
}

/* ── MINI ICONS ──────────────────────────────────────────────────────── */
function MiniIcon({type}){const s={width:38,height:22,display:"block",margin:"4px auto 0"};
if(type==="linearity")return <svg viewBox="0 0 36 22" style={s}><line x1="2" x2="34" y1="11" y2="11" stroke="#2B6CB0" strokeWidth=".7" strokeDasharray="2,1.5"/>{[[4,7],[7,15],[10,9],[14,16],[18,6],[22,15],[26,8],[30,14],[33,7]].map(([x,y],i)=><circle key={i} cx={x} cy={y} r="1.8" fill="#2B6CB0" opacity=".6"/>)}</svg>;
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
    <text x="150" y="14" textAnchor="middle" fontSize="11" fill={SB}>same slope estimate ± CI</text>
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
  // curve: saturing curve with a straight OLS line cutting through. Average slope hides changing effect
  if(type==="curve")return <svg viewBox="0 0 300 100" style={s}>
    <line x1="22" y1="84" x2="282" y2="84" stroke={BD}/>
    <line x1="22" y1="14" x2="22" y2="84" stroke={BD}/>
    {Array.from({length:13},(_,i)=>{const t=i/12;const x=30+i*21;const yBase=74-62*(1-Math.exp(-4.2*t));const y=yBase+(i%2?1.8:-2.2);return <circle key={i} cx={x} cy={y} r="3.2" fill={color} opacity=".78"/>;})}
    <line x1="24" y1="55" x2="282" y2="2" stroke={RF} strokeWidth="2.2"/>
    <text x="152" y="98" textAnchor="middle" fontSize="9.5" fill={SB}>straight line cannot follow the bend</text>
  </svg>;

  // dose_response: steep gains at low x, plateau at high x (study hours ↔ score style)
  if(type==="dose_response")return <svg viewBox="0 0 300 100" style={s}>
    <line x1="28" y1="80" x2="282" y2="80" stroke={BD}/>
    <line x1="28" y1="14" x2="28" y2="80" stroke={BD}/>
    <path d="M 28 78 C 72 74 118 52 278 22" fill="none" stroke={color} strokeWidth="2.4"/>
    <line x1="38" y1="76" x2="94" y2="54" stroke={SM} strokeWidth="1.6" strokeDasharray="3,3"/>
    <line x1="210" y1="28" x2="276" y2="24" stroke={SM} strokeWidth="1.6" strokeDasharray="3,3"/>
    <text x="62" y="96" textAnchor="middle" fontSize="9.5" fill={SB}>few hours</text>
    <text x="238" y="96" textAnchor="middle" fontSize="9.5" fill={SB}>many hours</text>
    <text x="14" y="48" textAnchor="middle" fontSize="10" fill={SB} transform="rotate(-90,14,48)">score</text>
  </svg>;

  // resid_curve: residuals make a pronounced U (systematic curvature after linear fit)
  if(type==="resid_curve")return <svg viewBox="0 0 300 100" style={s}>
    <line x1="22" y1="84" x2="282" y2="84" stroke={BD}/>
    <line x1="22" y1="14" x2="22" y2="84" stroke={BD}/>
    <line x1="22" x2="282" y1="50" y2="50" stroke={DH} strokeDasharray="4,3" opacity=".55"/>
    <path d="M 26 22 Q 152 94 274 22" fill="none" stroke={SM} strokeWidth="2.5"/>
    {Array.from({length:13},(_,i)=>{const x=34+i*20;const u=Math.cos((i/12)*Math.PI*2)*0.5+0.5;const y=21+(71-21)*(1-u)+(i*17%6-3);return <circle key={i} cx={x} cy={y} r="3" fill={color} opacity=".75"/>;})}
    <text x="14" y="48" textAnchor="middle" fontSize="10" fill={SB} transform="rotate(-90,14,48)">resid.</text>
    <text x="152" y="98" textAnchor="middle" fontSize="9.5" fill={SB}>fitted →</text>
  </svg>;

  // resid_good: random scatter, flat smoother
  if(type==="resid_good")return <svg viewBox="0 0 300 100" style={s}>
    <line x1="22" y1="84" x2="282" y2="84" stroke={BD}/>
    <line x1="22" y1="14" x2="22" y2="84" stroke={BD}/>
    <line x1="22" x2="282" y1="50" y2="50" stroke={DH} strokeDasharray="4,3" opacity=".55"/>
    <path d="M 22 50 Q 152 51 282 49" fill="none" stroke={SM} strokeWidth="2.5"/>
    {Array.from({length:13},(_,i)=>{const x=34+i*20;const y=50+([14,-10,-5,17,-14,6,-11,9,4,-13,-7,15,-5][i]);return <circle key={i} cx={x} cy={y} r="3" fill={color} opacity=".75"/>;})}
    <text x="14" y="48" textAnchor="middle" fontSize="10" fill={SB} transform="rotate(-90,14,48)">resid.</text>
    <text x="152" y="98" textAnchor="middle" fontSize="9.5" fill={SB}>fitted →</text>
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
  if(type==="outlier_types"){const NEU="#94a3b8";return <svg viewBox="0 0 300 110" style={s}>
    <rect x="6" y="22" width="92" height="64" fill="none" stroke={BD} rx="3"/>
    <line x1="10" y1="78" x2="94" y2="26" stroke={SM} strokeWidth="1.6"/>
    {[[16,72],[24,68],[30,64],[36,60],[42,56],[48,52],[54,48],[60,44],[66,40],[72,38],[80,32],[88,28]].map(([x,y],i)=><circle key={i} cx={x} cy={y} r="2.4" fill={NEU}/>)}
    <circle cx="40" cy="38" r="4" fill={RF}/>
    <text x="52" y="16" textAnchor="middle" fontSize="9.5" fill={SB} fontWeight="700">left panel</text>
    <rect x="104" y="22" width="92" height="64" fill="none" stroke={BD} rx="3"/>
    <line x1="108" y1="76" x2="192" y2="32" stroke={SM} strokeWidth="1.6"/>
    {[[114,72],[122,68],[128,64],[134,60],[140,56],[146,52],[152,48],[158,44],[164,42]].map(([x,y],i)=><circle key={i} cx={x} cy={y} r="2.4" fill={NEU}/>)}
    <circle cx="188" cy="34" r="4" fill={RF}/>
    <text x="150" y="16" textAnchor="middle" fontSize="9.5" fill={SB} fontWeight="700">middle panel</text>
    <rect x="202" y="22" width="92" height="64" fill="none" stroke={BD} rx="3"/>
    <line x1="206" y1="60" x2="290" y2="46" stroke={SM} strokeWidth="1.6"/>
    {[[210,68],[218,64],[226,60],[234,56],[242,52],[250,48],[258,44],[266,42],[272,40]].map(([x,y],i)=><circle key={i} cx={x} cy={y} r="2.4" fill={NEU}/>)}
    <circle cx="284" cy="74" r="5" fill={RF}/>
    <text x="248" y="16" textAnchor="middle" fontSize="9.5" fill={SB} fontWeight="700">right panel</text>
  </svg>;}

  // lev_resid: leverage vs std residual with Cook's contour, one influential point
  if(type==="lev_resid")return <svg viewBox="0 0 300 110" style={s}>
    <line x1="40" y1="86" x2="284" y2="86" stroke={BD}/>
    <line x1="40" y1="14" x2="40" y2="86" stroke={BD}/>
    <line x1="40" x2="284" y1="50" y2="50" stroke={DH} strokeDasharray="3,3" opacity=".5"/>
    <path d="M 40 18 Q 150 40 284 47" fill="none" stroke={SM} strokeDasharray="6,5" strokeWidth="1.4" opacity=".75"/>
    <path d="M 40 82 Q 150 60 284 53" fill="none" stroke={SM} strokeDasharray="6,5" strokeWidth="1.4" opacity=".75"/>
    {[[64,46],[78,56],[96,42],[112,54],[130,48],[148,52],[168,46],[188,48]].map(([x,y],i)=><circle key={i} cx={x} cy={y} r="2.8" fill="#94a3b8" opacity=".85"/>)}
    <circle cx="262" cy="40" r="5" fill={RF}/>
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
    {[8,12,7,14,9,11,8,15,10,13,9,11,46].map((h,i)=>{const x=42+i*18;const flagged=h>30;return <line key={i} x1={x} x2={x} y1={80} y2={80-h} stroke={flagged?RF:"#94a3b8"} strokeWidth="6" strokeLinecap="round"/>;})}
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
  if(type==="slope_shift"){const NEU="#94a3b8";return <svg viewBox="0 0 300 100" style={s}>
    <line x1="22" y1="84" x2="282" y2="84" stroke={BD}/>
    <line x1="22" y1="14" x2="22" y2="84" stroke={BD}/>
    {[[42,60],[58,55],[72,62],[88,52],[104,58],[120,50],[136,55],[152,48],[168,53],[184,46]].map(([x,y],i)=><circle key={i} cx={x} cy={y} r="3" fill={NEU} opacity=".9"/>)}
    <circle cx="266" cy="20" r="5.2" fill={RF} stroke="#fff" strokeWidth="1"/>
    <line x1="36" y1="58" x2="200" y2="48" stroke={DH} strokeWidth="2.2" strokeDasharray="5,4" opacity=".85"/>
    <line x1="36" y1="63" x2="278" y2="24" stroke={RF} strokeWidth="2.4"/>
    <rect x="32" y="10" width="120" height="30" rx="4" fill="#fff" stroke={BD}/>
    <line x1="40" y1="21" x2="62" y2="21" stroke={DH} strokeWidth="2" strokeDasharray="4,3"/>
    <text x="68" y="24" fontSize="9.5" fill={SB}>without red</text>
    <line x1="40" y1="34" x2="62" y2="34" stroke={RF} strokeWidth="2.2"/>
    <text x="68" y="37" fontSize="9.5" fill={RF} fontWeight="700">with red</text>
    <text x="152" y="98" textAnchor="middle" fontSize="9.5" fill={SB}>one point reshapes the slope</text>
  </svg>;}

  /* ── NORMALITY ─────────────────────────────────────────────── */
  // qq_heavy_tails: detrended Q-Q showing S-shape, heavy tails at both ends
  if(type==="qq_heavy_tails")return <svg viewBox="0 0 300 112" style={s}>
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
    <text x="155" y="105" textAnchor="middle" fontSize="9.5" fill={SB}>detrended Q-Q · normal quantile →</text>
  </svg>;

  // worm: detrended Q-Q with random scatter inside a confidence band
  if(type==="worm")return <svg viewBox="0 0 300 112" style={s}>
    <line x1="22" y1="84" x2="282" y2="84" stroke={BD}/>
    <line x1="22" y1="14" x2="22" y2="84" stroke={BD}/>
    <path d="M 30 36 Q 152 44 274 36 L 274 64 Q 152 56 30 64 Z" fill={SM} opacity=".13"/>
    <line x1="22" x2="282" y1="50" y2="50" stroke={SM} strokeWidth="2.2" opacity=".85"/>
    {[34,48,62,76,90,104,118,132,146,160,174,188,202,216,230,244,258,274].map((x,i)=>{
      const dev=[5,-4,3,-2,4,-3,1,-4,2,-1,3,-3,4,-2,1,-3,2,-4][i];
      return <circle key={i} cx={x} cy={50+dev} r="2.7" fill={color} opacity=".78"/>;
    })}
    <text x="155" y="105" textAnchor="middle" fontSize="9.5" fill={SB}>detrended Q-Q · normal quantile →</text>
  </svg>;

  // tails: detrended Q-Q with red dots at the ends, blue middle
  if(type==="tails")return <svg viewBox="0 0 300 112" style={s}>
    <line x1="22" y1="84" x2="282" y2="84" stroke={BD}/>
    <line x1="22" y1="14" x2="22" y2="84" stroke={BD}/>
    <line x1="22" x2="282" y1="50" y2="50" stroke={SM} strokeWidth="2.2" opacity=".85"/>
    {[34,56,80,104,128,154,180,206,230,254,278].map((x,i)=>{const offTail=i<2||i>8;const dev=offTail?(i<2?16:20):[2,-3,1,-2,3,-1,2][i-2];return <circle key={i} cx={x} cy={50+dev} r="3" fill={offTail?RF:color} opacity={offTail ? .85 : .7}/>;})}
    <text x="155" y="105" textAnchor="middle" fontSize="9.5" fill={SB}>detrended Q-Q · normal quantile →</text>
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
function DiagPlot({type,model,highlight,hiIdx,onHi,plotName,plotColor}){
  const brd=highlight?`2.5px solid ${highlight}`:`1.5px solid ${C.border}`;const bg=highlight?`${highlight}08`:"transparent";
  const titles={linearity:"Linearity",homogeneity:"Homogeneity of Variance",influential:"Influential Observations",normality:"Normality of Residuals"};
  const subtitles={linearity:"Reference line should be flat and horizontal",homogeneity:"Reference line should be flat and horizontal",influential:"Points should be inside the contour lines",normality:"Dots should fall along the line"};
  const plotLabel=plotName?<div style={{marginTop:6,paddingTop:6,borderTop:`1px solid ${C.border}`,fontSize:11.5,fontWeight:800,color:plotColor||C.sub,lineHeight:1.25}}>{plotName}</div>:null;
  if(!model)return <div style={{border:brd,borderRadius:10,padding:10,background:bg,minHeight:200,display:"flex",flexDirection:"column",justifyContent:"center"}}><span style={{fontSize:13,color:C.muted,textAlign:"center"}}>Need 3+ points</span>{plotLabel}</div>;
  const{fitted,residuals,stdRes,cooks,n}=model;
  const dot=(cx,cy,idx)=>{const isH=hiIdx===idx;return <g key={idx} style={{cursor:"pointer"}} onClick={e=>{e.stopPropagation();onHi?.(hiIdx===idx?null:idx);}}>{isH&&<circle cx={cx} cy={cy} r={9} fill={C.hiG}/>}<circle cx={cx} cy={cy} r={isH?5.5:3.5} fill={isH?C.hi:C.dot} stroke={isH?"#C53030":C.dotS} strokeWidth={isH?1.5:.6} opacity={(hiIdx!=null&&!isH) ? .3 : .8}/>{isH&&<text x={cx+8} y={cy-5} fontSize={9} fill={C.hi} fontWeight={700}>#{idx+1}</text>}</g>;};
  const wrap=(ch)=><div style={{border:brd,borderRadius:12,padding:"9px 9px 6px",background:highlight?bg:C.card,boxShadow:highlight?`0 0 0 3px ${highlight}12, 0 6px 18px rgba(0,0,0,.05)`:"0 2px 10px rgba(0,0,0,.04)",transition:"all .2s"}} onClick={()=>onHi?.(null)}><div style={{fontSize:13,fontWeight:800,color:C.text,paddingLeft:4,lineHeight:1.15}}>{titles[type]}</div><div style={{fontSize:10.5,color:C.sub,paddingLeft:4,marginBottom:2,lineHeight:1.15}}>{subtitles[type]}</div>{ch}{plotLabel}</div>;
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
function Scatter({points,setPoints,model,hiIdx,onHi,onEdit,xLabel="X",yLabel="Y"}){
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
    <text x={pw/2} y={ph+36} textAnchor="middle" fontSize={12} fill={C.text} fontWeight={600}>{xLabel}</text>
    <text x={-38} y={ph/2} textAnchor="middle" fontSize={12} fill={C.text} fontWeight={600} transform={`rotate(-90,-38,${ph/2})`}>{yLabel}</text>
    {model&&<line x1={sxx.fn(sxx.min)} y1={ph-syy.fn(model.b0+model.b1*sxx.min)} x2={sxx.fn(sxx.max)} y2={ph-syy.fn(model.b0+model.b1*sxx.max)} stroke={C.ref} strokeWidth={2.5} opacity={.55}/>}
    {points.map((p,i)=>{const isH=hiIdx===i;return <g key={i}>{isH&&<circle cx={sxx.fn(p.x)} cy={ph-syy.fn(p.y)} r={12} fill={C.hiG}/>}<circle cx={sxx.fn(p.x)} cy={ph-syy.fn(p.y)} r={drag===i?7:isH?6.5:5} fill={isH?C.hi:C.dot} stroke={isH?"#C53030":drag===i?C.ref:C.dotS} strokeWidth={isH||drag===i?2:1} opacity={(hiIdx!=null&&!isH) ? .25 : .85} style={{cursor:"grab"}} onPointerDown={e=>{e.stopPropagation();e.target.setPointerCapture(e.pointerId);lockedScale.current={sxx:liveSxx,syy:liveSyy};setDrag(i);moved.current=false;}} onClick={e=>{e.stopPropagation();if(!moved.current)onHi?.(hiIdx===i?null:i);}} onDoubleClick={e=>{e.stopPropagation();e.preventDefault();setPoints(pp=>pp.filter((_,j)=>j!==i));onEdit?.();}}/>{isH&&<text x={sxx.fn(p.x)+9} y={ph-syy.fn(p.y)-7} fontSize={11} fill={C.hi} fontWeight={700}>#{i+1}</text>}</g>;})}
  </g></svg>;
}

/* ── AUTOCORRELATION DIAGRAMS ────────────────────────────────────────── */
function AutocorrDiags(){
  // 28 days of ED asthma visits regressed on daily pollution. Truth: visits also
  // depend on day-of-week (more visits on weekends, when people aren't at work or
  // school). Pollution is roughly balanced across day-types, so the slope on
  // pollution stays approximately right; only the SE is fooled. Day-of-week acts
  // as a "cluster in time": weekend pairs and weekday runs share the same
  // residual sign, so the 28 rows carry far less information than 28 fresh ones.
  const[showColors,setShowColors]=useState(false);
  const data=[
    {d:1,wk:false,p:20,v:10},{d:2,wk:false,p:15,v:9}, {d:3,wk:false,p:25,v:10},
    {d:4,wk:false,p:30,v:11},{d:5,wk:false,p:18,v:9}, {d:6,wk:true, p:22,v:15},
    {d:7,wk:true, p:28,v:17},{d:8,wk:false,p:35,v:11},{d:9,wk:false,p:12,v:8},
    {d:10,wk:false,p:20,v:10},{d:11,wk:false,p:38,v:11},{d:12,wk:false,p:24,v:10},
    {d:13,wk:true, p:15,v:14},{d:14,wk:true, p:30,v:18},{d:15,wk:false,p:22,v:10},
    {d:16,wk:false,p:28,v:11},{d:17,wk:false,p:10,v:8}, {d:18,wk:false,p:20,v:10},
    {d:19,wk:false,p:32,v:11},{d:20,wk:true, p:18,v:14},{d:21,wk:true, p:25,v:16},
    {d:22,wk:false,p:15,v:9}, {d:23,wk:false,p:30,v:11},{d:24,wk:false,p:22,v:10},
    {d:25,wk:false,p:18,v:9}, {d:26,wk:false,p:26,v:10},{d:27,wk:true, p:20,v:14},
    {d:28,wk:true, p:28,v:17}
  ];
  const n=data.length;
  const mp=data.reduce((s,r)=>s+r.p,0)/n;
  const mv=data.reduce((s,r)=>s+r.v,0)/n;
  let sxx=0,sxy=0;
  data.forEach(r=>{const dp=r.p-mp;sxx+=dp*dp;sxy+=dp*(r.v-mv);});
  const slope=sxy/sxx, intercept=mv-slope*mp;
  // Coordinate mappings (200 x 120 viewBox)
  const minP=8,maxP=40,minV=6,maxV=20;
  const lx=p=>30+((p-minP)/(maxP-minP))*162;
  const ly=v=>96-((v-minV)/(maxV-minV))*78;
  const rx=d=>30+((d-1)/27)*162;
  const baseColor="#2E86AB", wkendColor="#C53030";
  const fillFor=wk=>showColors?(wk?wkendColor:baseColor):"#6B7B8A";
  // Vertical week-boundary guides for right panel (between Sun and Mon: between days 7&8, 14&15, 21&22)
  const weekBoundaries=[7.5,14.5,21.5];
  // Faint connecting line through points in time order
  const tsPath=data.map((r,i)=>`${i?"L":"M"}${rx(r.d)},${ly(r.v)}`).join(" ");
  return <div>
    <button onClick={()=>setShowColors(v=>!v)} style={{margin:"8px 0 10px",padding:"7px 12px",borderRadius:999,border:`1.5px solid ${baseColor}`,background:showColors?baseColor:"#fff",color:showColors?"#fff":baseColor,fontSize:12,fontWeight:800,fontFamily:"inherit",cursor:"pointer"}}>
      {showColors?"Hide":"Reveal"} weekend vs. weekday
    </button>
    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12,margin:"0 0 12px"}}>
    <div style={{background:C.bg,borderRadius:10,padding:12,border:`1.5px solid ${C.border}`}}>
      <div style={{fontSize:12,fontWeight:700,color:baseColor,marginBottom:6}}>Asthma visits vs. pollution</div>
      <svg viewBox="0 0 200 120" style={{width:"100%",height:"auto"}}>
        <line x1="30" x2="30" y1="15" y2="98" stroke={C.border} strokeWidth=".8"/>
        <line x1="30" x2="194" y1="98" y2="98" stroke={C.border} strokeWidth=".8"/>
        <line x1={lx(minP)} y1={ly(intercept+slope*minP)} x2={lx(maxP)} y2={ly(intercept+slope*maxP)} stroke="#444" strokeWidth="1.4" strokeDasharray="3,2.5" opacity=".55"/>
        {data.map(r=><circle key={r.d} cx={lx(r.p)} cy={ly(r.v)} r="3.6" fill={fillFor(r.wk)} stroke="#fff" strokeWidth=".8" opacity=".88"/>)}
        <text x="112" y="114" textAnchor="middle" fontSize="10" fill={C.sub}>Pollution</text>
        <text x="10" y="58" textAnchor="middle" fontSize="10" fill={C.sub} transform="rotate(-90,10,58)">ED visits</text>
      </svg>
      <div style={{fontSize:11,color:C.sub,lineHeight:1.5,marginTop:4}}>{showColors?<>Red weekend dots and blue weekday dots overlap on the pollution axis but cluster vertically. At any given pollution level, weekend visits sit higher than weekday visits.</>:<>Looks like 28 independent days with a clean positive slope.</>}</div>
    </div>
    <div style={{background:"#FFF5F5",borderRadius:10,padding:12,border:"1.5px solid #FED7D7"}}>
      <div style={{fontSize:12,fontWeight:700,color:wkendColor,marginBottom:6}}>Same visits, plotted by day</div>
      <svg viewBox="0 0 200 120" style={{width:"100%",height:"auto"}}>
        <line x1="30" x2="30" y1="15" y2="98" stroke={C.border} strokeWidth=".8"/>
        <line x1="30" x2="194" y1="98" y2="98" stroke={C.border} strokeWidth=".8"/>
        {weekBoundaries.map(b=><line key={b} x1={rx(b)} x2={rx(b)} y1="15" y2="98" stroke={C.border} strokeWidth=".5" strokeDasharray="2,2" opacity=".5"/>)}
        <line x1="30" x2="194" y1={ly(mv)} y2={ly(mv)} stroke="#444" strokeWidth="1" strokeDasharray="3,2.5" opacity=".5"/>
        <path d={tsPath} fill="none" stroke="#999" strokeWidth=".8" opacity=".4"/>
        {data.map(r=><circle key={r.d} cx={rx(r.d)} cy={ly(r.v)} r="3.3" fill={fillFor(r.wk)} stroke="#fff" strokeWidth=".7" opacity=".9"/>)}
        <text x="112" y="114" textAnchor="middle" fontSize="10" fill={C.sub}>Day of study</text>
        <text x="10" y="58" textAnchor="middle" fontSize="10" fill={C.sub} transform="rotate(-90,10,58)">ED visits</text>
      </svg>
      <div style={{fontSize:11,color:wkendColor,lineHeight:1.5,marginTop:4,fontWeight:600}}>{showColors?<>Each weekly block has 5 weekday dots below the mean and 2 weekend dots above. Adjacent days within each day-type lean the same way. They are clusters in time, not 28 independent observations.</>:<>Visits oscillate over time. Looks like noise until you reveal day-type.</>}</div>
    </div>
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

function ExogeneityDiags(){
  // 24 weeks across 3 temperature tiers. Within each tier, ice cream sales and
  // shark attacks have essentially no relationship; the strong overall slope is
  // entirely between-tier (driven by the hidden temperature variable).
  const[showColors,setShowColors]=useState(false);
  const data=[
    {ic:2,sa:2,t:1}, {ic:3,sa:3,t:1},{ic:2,sa:2,t:1}, {ic:4,sa:4,t:1},
    {ic:3,sa:3,t:1},{ic:4,sa:2,t:1}, {ic:2,sa:3,t:1},{ic:3,sa:4,t:1},
    {ic:5,sa:9,t:2},{ic:6,sa:11,t:2},{ic:7,sa:10,t:2},{ic:5,sa:13,t:2},
    {ic:6,sa:14,t:2},{ic:7,sa:9,t:2},{ic:6,sa:8,t:2},{ic:5,sa:12,t:2},
    {ic:9,sa:22,t:3},{ic:10,sa:26,t:3},{ic:11,sa:23,t:3},{ic:12,sa:25,t:3},
    {ic:9,sa:27,t:3},{ic:10,sa:20,t:3},{ic:11,sa:26,t:3},{ic:12,sa:21,t:3}
  ];
  const n=data.length;
  const mic=data.reduce((s,d)=>s+d.ic,0)/n;
  const msa=data.reduce((s,d)=>s+d.sa,0)/n;
  let sxx=0,sxy=0;
  data.forEach(d=>{const dx=d.ic-mic;sxx+=dx*dx;sxy+=dx*(d.sa-msa);});
  const slope=sxy/sxx, intercept=msa-slope*mic;
  const lx=ic=>30+((ic-1)/12)*162;
  const ly=sa=>96-(sa/30)*78;
  const baseColor="#8B6914";
  const tColors={1:"#3B82C4",2:"#C9A04A",3:"#C53030"};
  const tNames={1:"Cool",2:"Mild",3:"Hot"};
  const fillFor=t=>showColors?tColors[t]:"#A89968";
  return <div>
    <button onClick={()=>setShowColors(v=>!v)} style={{margin:"8px 0 10px",padding:"7px 12px",borderRadius:999,border:`1.5px solid ${baseColor}`,background:showColors?baseColor:"#fff",color:showColors?"#fff":baseColor,fontSize:12,fontWeight:800,fontFamily:"inherit",cursor:"pointer"}}>
      {showColors?"Hide":"Reveal"} the lurking variable
    </button>
    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12,margin:"0 0 12px"}}>
    <div style={{background:`${baseColor}10`,borderRadius:10,padding:12,border:`1.5px solid ${baseColor}55`}}>
      <div style={{fontSize:12,fontWeight:700,color:baseColor,marginBottom:6}}>The data</div>
      <svg viewBox="0 0 200 120" style={{width:"100%",height:"auto"}}>
        <line x1="30" x2="30" y1="15" y2="98" stroke={C.border} strokeWidth=".8"/>
        <line x1="30" x2="194" y1="98" y2="98" stroke={C.border} strokeWidth=".8"/>
        <line x1={lx(1)} y1={ly(intercept+slope*1)} x2={lx(13)} y2={ly(intercept+slope*13)} stroke="#444" strokeWidth="1.4" strokeDasharray="3,2.5" opacity=".55"/>
        {data.map((d,i)=><circle key={i} cx={lx(d.ic)} cy={ly(d.sa)} r="3.5" fill={fillFor(d.t)} stroke="#fff" strokeWidth=".7" opacity=".88"/>)}
        <text x="112" y="114" textAnchor="middle" fontSize="10" fill={C.sub}>Ice cream sales</text>
        <text x="10" y="58" textAnchor="middle" fontSize="10" fill={C.sub} transform="rotate(-90,10,58)">Shark attacks</text>
      </svg>
      {showColors && <div style={{display:"flex",gap:14,fontSize:11,color:C.sub,justifyContent:"center",marginTop:6}}>
        {[1,2,3].map(t=><span key={t} style={{display:"inline-flex",alignItems:"center",gap:4}}><span style={{display:"inline-block",width:9,height:9,borderRadius:"50%",background:tColors[t]}}/>{tNames[t]}</span>)}
      </div>}
      <div style={{fontSize:11,color:baseColor,lineHeight:1.5,marginTop:4,fontWeight:600}}>{showColors?<>Within each tier the slope is flat. The whole apparent effect is between tiers.</>:<>Shark attacks rise with ice cream sales.</>}</div>
    </div>
    <div style={{background:C.bg,borderRadius:10,padding:12,border:`1.5px solid ${C.border}`,display:"flex",flexDirection:"column"}}>
      <div style={{fontSize:12,fontWeight:700,color:baseColor,marginBottom:6}}>The hidden cause</div>
      {showColors ? <>
        <svg viewBox="0 0 200 120" style={{width:"100%",height:"auto"}}>
          <defs>
            <marker id="arrExoGold" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill={baseColor}/></marker>
            <marker id="arrExoGray" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#666"/></marker>
          </defs>
          <ellipse cx="100" cy="25" rx="44" ry="14" fill={baseColor} fillOpacity=".18" stroke={baseColor} strokeWidth="1.6"/>
          <text x="100" y="29" textAnchor="middle" fontSize="11" fill={baseColor} fontWeight="800">Temperature</text>
          <ellipse cx="42" cy="92" rx="34" ry="13" fill="#fff" stroke="#666" strokeWidth="1.4"/>
          <text x="42" y="95" textAnchor="middle" fontSize="9.5" fill="#333" fontWeight="700">Ice cream</text>
          <ellipse cx="158" cy="92" rx="34" ry="13" fill="#fff" stroke="#666" strokeWidth="1.4"/>
          <text x="158" y="95" textAnchor="middle" fontSize="9.5" fill="#333" fontWeight="700">Shark attacks</text>
          <path d="M 80 36 Q 60 60 47 80" fill="none" stroke={baseColor} strokeWidth="1.8" markerEnd="url(#arrExoGold)"/>
          <path d="M 120 36 Q 140 60 153 80" fill="none" stroke={baseColor} strokeWidth="1.8" markerEnd="url(#arrExoGold)"/>
          <line x1="77" y1="92" x2="125" y2="92" stroke="#666" strokeWidth="1.4" strokeDasharray="4,3" markerEnd="url(#arrExoGray)"/>
          <text x="101" y="88" textAnchor="middle" fontSize="11" fill="#C53030" fontWeight="800">?</text>
        </svg>
        <div style={{fontSize:11,color:C.sub,lineHeight:1.5,marginTop:4}}>Temperature drives both. OLS only sees the dashed arrow, whose slope absorbs the path through temperature.</div>
      </> : <div style={{flex:1,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",color:C.muted,fontSize:13,padding:"24px 12px",textAlign:"center"}}>
        <div style={{fontSize:54,fontWeight:300,opacity:.3,lineHeight:1}}>?</div>
        <div style={{marginTop:6}}>Click reveal to see what is driving both.</div>
      </div>}
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
    "# install.packages(c(\"ggplot2\", \"performance\", \"see\", \"qqplotr\"))",
    "library(ggplot2)",
    "library(performance)",
    "",
    `${safe} <- data.frame(`,
    `  x = c(${xs}),`,
    `  y = c(${ys})`,
    ")",
    "",
    `model <- lm(y ~ x, data = ${safe})`,
    "summary(model)",
    "",
    `ggplot(${safe}, aes(x = x, y = y)) +`,
    "  geom_point() +",
    "  geom_smooth(method = \"lm\", se = FALSE)",
    "",
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

function normalityDemoPoints(n=180,seedIn=20260429){
  let seed=seedIn>>>0;
  const rand=()=>{seed=(seed*1664525+1013904223)>>>0;return seed/4294967296;};
  const normal=()=>Math.sqrt(-2*Math.log(Math.max(rand(),1e-9)))*Math.cos(2*Math.PI*rand());
  const raw=Array.from({length:n},(_,i)=>{
    const x=1+(i/(n-1))*11+(rand()-.5)*.25;
    const noise=(normal()+normal())/Math.SQRT2;
    return{x,noise};
  });
  const m=raw.reduce((a,p)=>a+p.noise,0)/n;
  const sd=Math.sqrt(raw.reduce((a,p)=>a+(p.noise-m)**2,0)/Math.max(n-1,1))||1;
  return raw.map(p=>{
    const z=(p.noise-m)/sd;
    return{x:+p.x.toFixed(2),y:+(18+2.4*p.x+z*2.6).toFixed(2)};
  });
}

function NormalityWalkthrough({color="#6B46C1"}){
  const[demoPoints,setDemoPoints]=useState(()=>normalityDemoPoints());
  const[hiIdx,setHiIdx]=useState(null);
  const[custom,setCustom]=useState(false);
  const model=useMemo(()=>ols(demoPoints),[demoPoints]);
  if(!model)return null;
  const res=model.residuals||[],std=model.studentRes||model.stdRes||[];
  if(res.length<3||std.length<3)return null;
  const idx=res.map((_,i)=>i+1);
  const qq=qqPts(std),worm=qqWorm(std);
  const mean=std.reduce((a,v)=>a+v,0)/std.length;
  const sd=Math.sqrt(std.reduce((a,v)=>a+(v-mean)**2,0)/Math.max(std.length-1,1));
  const binCount=Math.min(10,Math.max(6,Math.round(Math.sqrt(std.length))));
  let mn=Math.min(...std),mx=Math.max(...std);
  const ext=Math.max(Math.abs(mn),Math.abs(mx),3);
  mn=-ext;mx=ext;
  const bw=(mx-mn)/binCount;
  const bins=Array.from({length:binCount},(_,i)=>({lo:mn+i*bw,hi:mn+(i+1)*bw,count:0}));
  std.forEach(r=>{const i=Math.min(binCount-1,Math.max(0,Math.floor((r-mn)/bw)));bins[i].count+=1;});
  const card=(title,body,plot)=><div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:10,boxShadow:"0 2px 10px rgba(0,0,0,.035)"}}>
    <div style={{fontSize:13,fontWeight:800,color,marginBottom:3}}>{title}</div>
    <div style={{fontSize:11.5,color:C.sub,lineHeight:1.4,marginBottom:6}}>{body}</div>
    {plot}
  </div>;
  return <div style={{background:"#F7F2FF",border:"1.5px solid #D8C8F0",borderRadius:14,padding:16,marginTop:16}}>
    <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",gap:12,marginBottom:10}}>
      <div>
        <div style={{fontSize:16,fontWeight:800,color,marginBottom:4}}>Want more on Q-Q plots?</div>
        <p style={{...pa,fontSize:13}}>This larger demo starts with roughly normal residuals so the histogram and Q-Q plots are easier to read. Drag the base scatterplot points to see all four residual views update.</p>
      </div>
      <button onClick={()=>{setDemoPoints(normalityDemoPoints(undefined,(Math.random()*2**31)>>>0));setHiIdx(null);setCustom(false);}} style={{padding:"6px 10px",borderRadius:8,border:`1.5px solid ${color}`,background:custom?color:"#fff",color:custom?"#fff":color,fontSize:12,fontWeight:800,cursor:"pointer",fontFamily:"inherit",whiteSpace:"nowrap"}}>New random sample</button>
    </div>
    <div style={{display:"grid",gridTemplateColumns:"minmax(500px, 1.45fr) minmax(220px, .55fr)",gap:12,alignItems:"start",marginBottom:12}}>
      <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:10,boxShadow:"0 2px 10px rgba(0,0,0,.035)"}}>
        <div style={{fontSize:13,fontWeight:800,color,marginBottom:3}}>Base scatterplot</div>
        <div style={{fontSize:11.5,color:C.sub,lineHeight:1.4,marginBottom:6}}>The residuals below come from this fitted line. Drag, add, or remove points.</div>
        <Scatter points={demoPoints} setPoints={setDemoPoints} model={model} hiIdx={hiIdx} onHi={setHiIdx} onEdit={()=>setCustom(true)} xLabel="Predictor" yLabel="Outcome"/>
      </div>
      <div style={{background:"#FFFFFF",border:`2px solid ${color}55`,borderRadius:12,padding:14,boxShadow:`0 5px 16px ${color}18`}}>
        <div style={{fontSize:14,fontWeight:900,color,marginBottom:10}}>Helpful Q-Q Plot Videos</div>
        <div style={{display:"flex",flexDirection:"column",gap:9}}>
          <a href="https://www.youtube.com/watch?v=okjYjClSjOg" target="_blank" rel="noopener noreferrer" style={{display:"block",padding:"10px 12px",borderRadius:10,background:`${color}12`,border:`1.5px solid ${color}35`,color,textDecoration:"none",fontSize:13,fontWeight:900,lineHeight:1.25}}>StatQuest: Q-Q plots</a>
          <a href="https://www.youtube.com/watch?v=X9_ISJ0YpGw" target="_blank" rel="noopener noreferrer" style={{display:"block",padding:"10px 12px",borderRadius:10,background:`${color}12`,border:`1.5px solid ${color}35`,color,textDecoration:"none",fontSize:13,fontWeight:900,lineHeight:1.25}}>JB Statistics: Q-Q plots</a>
        </div>
      </div>
    </div>
    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
      {card("1. Residuals", "Start with the prediction errors: actual minus fitted.", <CF xs={idx} ys={[...res,0]} xL="Observation" yL="Residual" w={330} h={220}>{(sx,sy,w)=><>{<line x1={0} x2={w} y1={sy(0)} y2={sy(0)} stroke={C.dash} strokeWidth={1} strokeDasharray="5,5" opacity={.55}/>} {res.map((r,i)=><circle key={i} cx={sx(i+1)} cy={sy(r)} r="3.5" fill={color} opacity=".75"/>)}</>}</CF>)}
      {card("2. Histogram", "Standardize the residuals (z-scores) so the spread is comparable to a normal distribution. The histogram should look bell-shaped, centered at 0.", <CF xs={[mn,mx]} ys={[0,...bins.map(b=>b.count)]} xL="Standardized residual" yL="Count" w={330} h={220}>{(sx,sy,_,ph)=><>{bins.map((b,i)=><rect key={i} x={sx(b.lo)+1} y={sy(b.count)} width={Math.max(1,sx(b.hi)-sx(b.lo)-2)} height={ph-sy(b.count)} fill={color} opacity=".45" stroke={color} strokeWidth=".7"/>)}</>}</CF>)}
      {card("3. Normal Q-Q plot", "Sort the standardized residuals and compare them with where normal residuals would fall. The y-axis matches the histogram's x-axis.", <CF xs={qq.map(q=>q.th)} ys={[...qq.map(q=>q.sa),...qq.map(q=>mean+sd*q.th)]} xL="Theoretical normal quantile" yL="Standardized residual" w={330} h={220}>{(sx,sy)=><>{<line x1={sx(Math.min(...qq.map(q=>q.th)))} x2={sx(Math.max(...qq.map(q=>q.th)))} y1={sy(mean+sd*Math.min(...qq.map(q=>q.th)))} y2={sy(mean+sd*Math.max(...qq.map(q=>q.th)))} stroke={C.smooth} strokeWidth={2} opacity=".65"/>}{qq.map((q,i)=><circle key={i} cx={sx(q.th)} cy={sy(q.sa)} r="3.3" fill={color} opacity=".75"/>)}</>}</CF>)}
      {card("4. Detrended Q-Q plot", "Flatten the Q-Q line to zero so departures are easier to see.", <CF xs={worm.map(q=>q.th)} ys={[...worm.map(q=>q.dev),...worm.map(q=>q.lo),...worm.map(q=>q.hi),0]} xL="Theoretical normal quantile" yL="Deviation from line" w={330} h={220}>{(sx,sy,w)=>{
        const top=worm.map(q=>`L${sx(q.th)},${sy(q.hi)}`).join(" ");
        const bot=[...worm].reverse().map(q=>`L${sx(q.th)},${sy(q.lo)}`).join(" ");
        return <>{worm.length>1&&<path d={`M${sx(worm[0].th)},${sy(worm[0].hi)} ${top.slice(1)} ${bot} Z`} fill={color} opacity=".12" stroke="none"/>}<line x1={0} x2={w} y1={sy(0)} y2={sy(0)} stroke={C.smooth} strokeWidth={2} opacity=".75"/>{worm.map((q,i)=><circle key={i} cx={sx(q.th)} cy={sy(q.dev)} r="3.3" fill={color} opacity=".75"/>)}</>;
      }}</CF>)}
    </div>
  </div>;
}

const INFO={
sample:()=><div style={{display:"flex",flexDirection:"column",gap:16}}>
  <div><Hd>The key rule</Hd><p style={pa}>A common rule of thumb is to aim for at least <b>10 observations per parameter</b> you are estimating.</p></div>
  <div><Hd>What counts as a parameter?</Hd>
    <p style={pa}>Each coefficient counts as a parameter. Simple linear regression has two (intercept and slope), so the rule suggests at least <b>20</b> observations.</p>
    <p style={{...pa,marginTop:10}}>Add a second predictor and you have three parameters, pushing the minimum to <b>30</b>.</p>
  </div>
  <div><Hd>Why it matters</Hd>
    <p style={pa}>Small samples produce noisier estimates and weaker diagnostics. Two issues come up again and again:</p>
    <ul style={ul}>
      <li><b>You cannot really check your assumptions.</b> Residual plots need enough points to be interpretable.</li>
      <li><b>Individual observations dominate.</b> One unusual point can drag the line, and with small <i>n</i> you may not notice.</li>
    </ul>
  </div>
</div>,

uncorrelated:()=><div style={{display:"flex",flexDirection:"column",gap:18}}>
  <div><Hd>What it means</Hd><p style={pa}>Each observation's prediction error should carry no information about any other. If knowing one residual helps you guess the next, the errors are correlated.</p></div>

  <div style={{background:"#F5F0FF",border:"1.5px solid #D0C0E8",borderRadius:12,padding:16,display:"flex",flexDirection:"column",gap:14}}>
    <div style={{fontSize:16,fontWeight:700,color:"#6B46C1"}}>Violation type 1: Clustered data</div>
    <div><p style={pa}>Observations in the same group (hospital, school, neighborhood) tend to resemble each other. Their errors are correlated.</p>
    <p style={{...pa,marginTop:6}}><b>Example:</b> you regress patient satisfaction on minutes with doctor, using patients from 10 clinics. Same-clinic patients share staff, scheduling, and crowding, so their residuals will be correlated.</p></div>
    <ClusteredDiags/>
    <div><Hd>What goes wrong</Hd><ul style={ul}><li><b>Standard errors tend to be too small.</b> The model overestimates how much independent information the data contain.</li><li><b>Confidence intervals are typically too narrow</b> and <b>p-values typically too small.</b></li></ul></div>
    <div><Hd>How to assess</Hd><FixList items={[
      {text:"**Think about data collection.** Are observations grouped by site, school, provider, family?"},
      {text:"Compute the **intraclass correlation coefficient (ICC)** to measure how much variance is between vs. within clusters.",
        links:[{title:"performance::icc: ICC for mixed models",short:"icc()",url:"https://easystats.github.io/performance/reference/icc.html"},{title:"R-bloggers: ICC in R quick guide",short:"R-bloggers",url:"https://www.r-bloggers.com/2021/06/intraclass-correlation-coefficient-in-r-quick-guide/"}]}
    ]}/></div>
    <div><Hd>How to fix</Hd><FixList items={[
      {text:"Fit a **multilevel or mixed-effects model** that accounts for the grouping.",
        links:[{title:"Mixed Models with R (Michael Clark)",short:"m-clark",url:"https://m-clark.github.io/mixed-models-with-R/"},{title:"lme4 vignette: Fitting linear mixed-effects models",short:"lme4 vignette",url:"https://cran.r-project.org/web/packages/lme4/vignettes/lmer.pdf"}]},
      {text:"Use **cluster-robust standard errors** as a simpler option when a full multilevel model is not feasible.",
        links:[{title:"Miratrix: MLM and cluster-robust SE",short:"Miratrix",url:"https://lmiratrix.github.io/MLM/robust_mlm.html"},{title:"easystats: Robust SE, CIs and p-values",short:"easystats",url:"https://easystats.github.io/parameters/articles/model_parameters_robust.html"}]},
      {text:"At minimum, **acknowledge** the clustering as a limitation when reporting results."}
    ]}/></div>
  </div>

  <div style={{background:"#F0F8FF",border:"1.5px solid #B8D8F0",borderRadius:12,padding:16,display:"flex",flexDirection:"column",gap:14}}>
    <div style={{fontSize:16,fontWeight:700,color:"#2E86AB"}}>Violation type 2: Autocorrelation (time-ordered data)</div>
    <div><Hd>What is happening</Hd><p style={pa}>When data is collected over time, nearby observations often have similar residuals.</p>
    <p style={{...pa,marginTop:6}}><b>Example:</b> 28 days of emergency-department asthma visits regressed on daily pollution. The slope is real, but visits also rise on weekends, and day-of-week isn't in the model. Weekends and weekdays show up as runs of similar residuals. The slope stays roughly right, but the standard errors are computed as if every day were independent, when the 28 rows behave more like ~10 effective observations.</p></div>
    <AutocorrDiags/>
    <div><Hd>What goes wrong</Hd><ul style={ul}><li><b>Standard errors tend to be too small.</b> The model overestimates how much independent information the data contain.</li><li><b>Confidence intervals are typically too narrow</b> and <b>p-values typically too small.</b></li></ul></div>
    <div><Hd>How to assess</Hd><FixList items={[
      {text:"**Plot residuals in collection order** (by time, by sequence). Look for runs or waves.",
        links:[{title:"R-bloggers: Residual plots and assumption checking",short:"R-bloggers",url:"https://www.r-bloggers.com/2020/10/residual-plots-and-assumption-checking/"},{title:"forecast::checkresiduals (acf + Ljung-Box)",short:"checkresiduals",url:"https://pkg.robjhyndman.com/forecast/reference/checkresiduals.html"}]},
      {text:"**Durbin–Watson test** for first-order autocorrelation.",
        links:[{title:"Statology: Durbin-Watson test in R",short:"Statology",url:"https://www.statology.org/durbin-watson-test-r/"},{title:"lmtest::dwtest reference",short:"dwtest",url:"https://rdrr.io/cran/lmtest/man/dwtest.html"}]}
    ]}/></div>
    <div><Hd>How to fix</Hd><FixList items={[
      {text:"Use a **time-series model** (for example ARIMA errors or GLS with an autocorrelation structure).",
        links:[{title:"Fish-Forecast: Multivariate regression with ARMA errors",short:"Fish-Forecast",url:"https://fish-forecast.github.io/Fish-Forecast-Bookdown/6-2-multivariate-linear-regression-with-arma-errors.html"},{title:"DataScience+: Regression with autocorrelated errors",short:"DataScience+",url:"https://datascienceplus.com/regression-model-with-auto-correlated-errors-part-2-the-models/"}]},
      {text:"Add **lagged variables** as predictors when dynamics matter.",
        links:[{title:"dynlm package (CRAN)",short:"dynlm",url:"https://cran.r-project.org/web/packages/dynlm/dynlm.pdf"},{title:"Time Series Analysis With R, Ch. 9: Regression",short:"TSAwR Ch.9",url:"https://nicolarighetti.github.io/Time-Series-Analysis-With-R/regression.html"}]},
      {text:"Use **Newey–West or HAC standard errors** for inference with serially correlated errors.",
        links:[{title:"Econometrics with R: HAC standard errors",short:"EWR",url:"https://www.econometrics-with-r.org/15.4-hac-standard-errors.html"},{title:"sandwich package vignette (CRAN)",short:"sandwich",url:"https://cran.r-project.org/web/packages/sandwich/vignettes/sandwich.pdf"}]}
    ]}/></div>
  </div>
</div>,

multicollinearity:()=><div style={{display:"flex",flexDirection:"column",gap:14}}>
  <div style={{background:"#FFF8E1",border:"1.5px solid #E8D080",borderRadius:8,padding:"12px 16px",fontSize:14,color:"#6B5B1F",lineHeight:1.6}}>
    This applies to <b>multiple regression only</b>. With one predictor, it cannot occur. Included here for the mnemonic.
  </div>
  <div><Hd>What it means</Hd><p style={pa}>When two or more predictors carry largely the same information, the model cannot tell their individual effects apart, so each slope becomes unstable.</p></div>
  <div><Hd>Two types</Hd><ul style={ul}>
    <li><b>Perfect multicollinearity:</b> one predictor is an exact linear function of another (e.g., temperature in Celsius and in Fahrenheit). The model cannot be estimated at all. Software will drop one variable or throw an error.</li>
    <li><b>Imperfect multicollinearity</b> (the practical concern): predictors are highly but not perfectly correlated; for example, <b>years of education</b> and <b>household income</b> predicting health outcomes. Software can fit the model, but the slopes for those predictors are hard to pin down separately.</li>
  </ul></div>
  <div><Hd>What goes wrong</Hd><p style={pa}>With <b>perfect multicollinearity</b>, you <b>cannot obtain unique coefficient estimates</b>. With strong <b>imperfect multicollinearity</b>, individual slopes are <b>hard to interpret</b> (they can shift a lot with small changes in the data), and their <b>standard errors are inflated</b>, so tests and intervals for those coefficients are <b>unreliable</b>.</p></div>
  <div><Hd>How to assess</Hd><FixList items={[
    {text:"**VIF (variance inflation factor)** for each predictor. Values above 5 or 10 are commonly treated as a problem.",
      links:[{title:"Statology: Calculate VIF in R",short:"Statology",url:"https://www.statology.org/variance-inflation-factor-r/"},{title:"STHDA: Multicollinearity essentials and VIF in R",short:"STHDA",url:"https://www.sthda.com/english/articles/39-regression-model-diagnostics/160-multicollinearity-essentials-and-vif-in-r/"}]},
    {text:"**Correlation matrix** (or heatmap) of the predictors.",
      links:[{title:"STHDA: Correlation matrix in R",short:"STHDA",url:"http://www.sthda.com/english/wiki/correlation-matrix-a-quick-start-guide-to-analyze-format-and-visualize-a-correlation-matrix-using-r-software"},{title:"corrplot package vignette (CRAN)",short:"corrplot",url:"https://cran.r-project.org/web/packages/corrplot/vignettes/corrplot-intro.html"}]}
  ]}/></div>
</div>,

exogeneity:()=><div style={{display:"flex",flexDirection:"column",gap:14}}>
  <div style={{background:"#FFF8E1",border:"1.5px solid #E8D080",borderRadius:8,padding:"12px 16px",fontSize:14,color:"#6B5B1F",lineHeight:1.6}}>
    <ul style={{...ul,marginTop:0,marginBottom:0,color:"inherit"}}>
      <li><b>Exogeneity is different</b> from the usual “does the model fit?” questions. It asks whether the slope is actually <b>causal</b>: would changing X really move Y by about that much?</li>
      <li>If exogeneity fails, you can still describe and predict with the slope. You just shouldn’t treat it as a causal effect.</li>
    </ul>
  </div>
  <div><Hd>Example</Hd><p style={pa}>Across summer weeks, weekly <b>ice cream sales</b> and weekly <b>shark attacks</b> rise and fall together. Of course ice cream does not summon sharks. <b>Air temperature</b> drives both.</p></div>
  <ExogeneityDiags/>
  <div><Hd>How to assess</Hd><p style={pa}>No residual plot can certify exogeneity. Ask: could some third variable drive both X and Y?</p></div>
  <div><Hd>How to fix</Hd><FixList items={[
    {text:"**Randomized experiments** solve it by design: random assignment breaks the link between predictor and unobserved confounders."},
    {text:"**Add control variables** for plausible confounders in a multiple regression.",
      links:[{title:"R-bloggers: Controlling for confounders in R",short:"R-bloggers",url:"https://www.r-bloggers.com/2022/01/simple-examples-to-understand-what-confounders-colliders-mediators-and-moderators-are-and-how-to-control-for-variables-in-r-with-regression-and-propensity-score-matching/"},{title:"Andrew Heiss: DAGs and adjustment sets in R",short:"Heiss",url:"https://evalf20.classes.andrewheiss.com/example/dags/"}]},
    {text:"Use **instrumental-variables regression** when a valid instrument is available.",
      links:[{title:"Econometrics with R, Ch. 12: IV regression",short:"EWR",url:"https://www.econometrics-with-r.org/12-ivr.html"},{title:"Andrew Heiss: Instrumental variables in R",short:"Heiss",url:"https://evalf20.classes.andrewheiss.com/example/iv/"}]},
    {text:"At minimum: **draw a DAG** and ask whether a third variable drives both X and Y.",
      links:[{title:"ggdag: Drawing causal DAGs in R",short:"ggdag",url:"https://r-causal.github.io/ggdag/"},{title:"Causal Inference in R, Ch. 4: DAGs",short:"R-causal",url:"https://www.r-causal.org/chapters/04-dags"}]}
  ]}/></div>
  <p style={{...pa,marginTop:4,color:C.muted,fontStyle:"italic"}}>We will return to this with multiple regression and causal inference.</p>
</div>,
};

function InfoPanel({item}){const Content=INFO[item.key];return <div style={{"--accent":item.color,background:C.card,border:`1.5px solid ${C.border}`,borderTop:`7px solid ${item.color}`,borderRadius:16,padding:24,maxWidth:840,boxShadow:C.shadow}}>
  <div style={{fontSize:22,fontWeight:800,color:item.color,marginBottom:2}}>{item.label}</div>
  <div style={{fontSize:14,fontWeight:500,color:C.sub,marginBottom:16,fontStyle:"italic"}}>{item.summary}</div>
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
  const diagTypes=["linearity","homogeneity","influential","normality"];
  const diagMeta=Object.fromEntries(diagTypes.map(type=>{const d=SUNSHINE.find(s=>s.diagKey===type);const names=d?.readingPlotNames||[];const name=d?.plotCaption??(names[1]?`${names[0]} (${names[1]})`:names[0]);return[type,{name,color:d?.color||C.sub}];}));
  const pill=k=>({padding:"8px 16px",borderRadius:20,fontSize:13,fontWeight:active===k?700:500,border:`1.5px solid ${active===k?item.color:C.border}`,background:active===k?item.colorSoft:"transparent",color:active===k?item.color:C.muted,cursor:"pointer",transition:"all .15s",whiteSpace:"nowrap"});
  const smBtn={padding:"4px 10px",borderRadius:6,border:`1px solid ${C.border}`,background:"transparent",fontSize:11,color:C.muted,cursor:"pointer",fontFamily:"inherit"};

  return <div style={{display:"flex",flexDirection:"column",gap:14}}>
    <div style={{background:C.card,border:`1.5px solid ${C.border}`,borderTop:`7px solid ${item.color}`,borderRadius:16,padding:"16px 22px",boxShadow:C.shadow}}>
      <div style={{fontSize:22,fontWeight:800,color:item.color,marginBottom:2}}>
        {item.label} {item.labelParen&&<span style={{fontWeight:500,fontSize:15,color:C.sub}}>{item.labelParen}</span>}
      </div>
      <div style={{fontSize:14,fontWeight:500,color:C.sub,marginBottom:14,fontStyle:"italic"}}>{item.summary}</div>
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
    <div style={{display:"grid",gridTemplateColumns:"minmax(520px, 1.1fr) minmax(470px, 1fr)",gap:14,alignItems:"start",minWidth:1000}}>
      <div style={{background:C.card,border:`1.5px solid ${C.border}`,borderRadius:16,padding:14,boxShadow:C.shadow}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:4,flexWrap:"wrap",gap:6}}>
          <span style={{fontSize:15,fontWeight:700,color:C.text}}>Scatter Plot</span>
          <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
            <button onClick={handleDownload} disabled={!points.length} title="Download the current points as a CSV" style={smBtn}>Download CSV</button>
            <button onClick={handleCopyR} disabled={!points.length} title="Copy R code: data.frame, lm(y ~ x), ggplot scatter + line, and check_model()" style={smBtn}>{copied?"Copied!":"Copy R code"}</button>
            <button onClick={()=>{setPoints([]);setCustom(true);setHiIdx(null);}} style={smBtn}>Clear all</button>
          </div>
        </div>
        <div style={{display:"flex",gap:10,marginBottom:8,fontSize:12,color:C.sub,flexWrap:"wrap"}}>
          <span><b>+</b> Double-click to add</span>
          <span><b>&harr;</b> Drag to move</span>
          <span><b>&times;</b> Double-click point to remove</span>
          <span><b>&bull;</b> Click point to track across plots</span>
        </div>
        <Scatter points={points} setPoints={setPoints} model={model} hiIdx={hiIdx} onHi={setHiIdx} onEdit={onEdit} xLabel={ds?.xLabel} yLabel={ds?.yLabel}/>
        {model&&<div style={{marginTop:8,fontFamily:"'JetBrains Mono',monospace",fontSize:14,color:C.sub,padding:"6px 12px",background:C.bg,borderRadius:6}}>
          y&#770; = {model.b0.toFixed(2)} + {model.b1.toFixed(2)}x &nbsp;&nbsp;&nbsp; n = {points.length}
        </div>}
      </div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
        {diagTypes.map(dt=><DiagPlot key={dt} type={dt} model={model} highlight={dt===item.diagKey?item.color:null} hiIdx={hiIdx} onHi={setHiIdx} plotName={diagMeta[dt].name} plotColor={diagMeta[dt].color}/>)}
      </div>
    </div>
    <div style={{"--accent":item.color,background:C.card,border:`1.5px solid ${C.border}`,borderTop:`7px solid ${item.color}`,borderRadius:16,padding:22,boxShadow:C.shadow}}>
      <Hd>What it means</Hd><p style={pa}>{item.explanation}</p>
      <div style={{marginTop:14}}><Hd>Reading the plots:</Hd>{item.readingPlotNames?.length?<p style={{...pa,marginTop:2,marginBottom:10,fontSize:14,fontWeight:400,lineHeight:1.45,color:item.color}}>{item.readingPlotNames[0]}{item.readingPlotNames[1]?<span style={{color:C.muted,fontStyle:"italic"}}> (also called {item.readingPlotNames[1]})</span>:null}</p>:null}<div style={{...pa,display:"flex",flexDirection:"column",gap:4,marginTop:0}}>{item.plotGuide.split("\n").map((line,i)=><div key={i}><MdBold text={line}/></div>)}</div></div>
      <div style={{marginTop:14}}><Hd>Effect on your results</Hd><WhatBreaks data={item.whatBreaks}/></div>
      <div style={{marginTop:14}}><Hd>Formal test</Hd><FixList items={item.formalTestList}/></div>
      <div style={{marginTop:14}}><Hd>How to fix</Hd><FixList items={item.howToFixList}/></div>
      {item.key==="normality"&&<Quiz questions={MCQ[item.key]} color={item.color}/>}
      {item.key==="normality"&&<NormalityWalkthrough color={item.color}/>}
      {item.key!=="normality"&&<Quiz questions={MCQ[item.key]} color={item.color}/>}
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
      <div style={{display:"grid",gridTemplateColumns:"repeat(8, minmax(0, 1fr))",gap:5,marginBottom:20}}>
        <div style={{gridColumn:"1 / 3"}}/>
        <div style={{gridColumn:"3 / span 5",textAlign:"center",fontSize:11,fontWeight:700,color:C.teal,lineHeight:1.35,paddingBottom:2,paddingLeft:18,paddingRight:18}}>
          Checkable with{" "}
          <code style={{background:C.tealBg,padding:"1px 6px",borderRadius:4,color:C.tealDark,fontFamily:"JetBrains Mono, monospace",fontSize:10,fontWeight:600}}>performance::check_model(model)</code>
        </div>
        <div style={{gridColumn:"8 / 9"}}/>
        <div style={{gridColumn:"1 / 3"}}/>
        <div style={{gridColumn:"3 / span 5",height:8,borderTop:`2px solid ${C.teal}`,borderLeft:`2px solid ${C.teal}`,borderRight:`2px solid ${C.teal}`,borderTopLeftRadius:8,borderTopRightRadius:8,alignSelf:"end",marginLeft:18,marginRight:18}}/>
        <div style={{gridColumn:"8 / 9"}}/>
        {SUNSHINE.map(item=>{const isA=selected===item.key;const isD=item.type==="diagnostic";return <button key={item.key} onClick={()=>handleSel(item.key)} style={{display:"flex",flexDirection:"column",alignItems:"center",padding:"10px 6px 6px",borderRadius:12,minWidth:0,border:`2.5px solid ${isA?item.color:C.border}`,background:isA?item.colorSoft:C.card,cursor:"pointer",transition:"all .2s",position:"relative",boxShadow:isA?`0 3px 16px ${item.color}25`:"0 1px 3px #0001",transform:isA?"translateY(-2px)":"none"}}>
          <span style={{fontWeight:800,fontSize:27,lineHeight:1,color:isA?item.color:"#A09A90"}}>{item.letter}</span>
          <span style={{fontSize:8.5,color:isA?item.color:C.muted,textAlign:"center",lineHeight:1.2,marginTop:3,fontWeight:isA?800:600,maxWidth:76}}>{item.label}{item.labelParen?` ${item.labelParen}`:""}</span>
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
          <p style={{margin:"0 0 8px"}}>Reading these plots is a learned skill, and two analysts can disagree on the same plot. <b>Formal tests</b> (Breusch-Pagan, Shapiro-Wilk, etc.) sound more objective but are not a clean fix: they over-reject in large samples (flagging trivial departures) and miss real violations in small ones. Use them as a supplement to the plots, not a replacement.</p>
          <p style={{margin:0}}>If regression is central to your analysis, <b>report your checks</b> (often in supplementary materials) so readers can judge for themselves.</p>
        </div>}
      </div>}
    </div>
  </div>;
}

const rootEl = document.getElementById("root");
if (rootEl) createRoot(rootEl).render(<App />);
