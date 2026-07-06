#!/usr/bin/env python3
"""Build the Codex HTML version of the donut-audit story deck."""
from pathlib import Path

HERE = Path(__file__).parent

HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Donut audit t-test story</title>
  <style>
    :root {
      --gold: #d4af33;
      --mist: #c4d8db;
      --teal: #035f6c;
      --alarm: #b5451f;
      --brown: #7a6212;
      --ink: #073b42;
      --stage: #0d2b30;
      --paper: #ffffff;
      --soft: #f2f7f7;
      --warm: #fbeecb;
      --coral: #e58b73;
      --violet: #8c7ab8;
      --sans: Inter, "Helvetica Neue", Arial, sans-serif;
      --mono: "JetBrains Mono", "SF Mono", Menlo, monospace;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      height: 100%;
      overflow: hidden;
      background: var(--stage);
      color: var(--ink);
      font-family: var(--sans);
    }
    #stage {
      position: fixed;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 42px 14px 34px;
    }
    #deck {
      position: relative;
      width: min(96vw, 163.56vh);
      aspect-ratio: 16 / 9;
      overflow: hidden;
      border-radius: 10px;
      background: var(--paper);
      box-shadow: 0 18px 60px rgba(0,0,0,.55);
    }
    .slide {
      position: absolute;
      inset: 0;
      display: none;
      grid-template-rows: 12px auto 1fr 12px;
      background: var(--paper);
    }
    .slide.active { display: grid; }
    .chrome-top { background: var(--gold); }
    .chrome-bottom { background: var(--teal); }
    .slide-head {
      padding: 16px 34px 6px;
    }
    .kicker {
      margin-bottom: 4px;
      color: var(--brown);
      font-size: 12px;
      font-weight: 850;
      letter-spacing: 1.9px;
      text-transform: uppercase;
    }
    h1 {
      margin: 0;
      color: var(--teal);
      font-size: clamp(23px, 2.22vw, 34px);
      line-height: 1.04;
      font-weight: 850;
      letter-spacing: 0;
    }
    .title-line {
      width: min(420px, 38%);
      height: 5px;
      margin-top: 8px;
      border-radius: 4px;
      background: var(--gold);
    }
    .slide-body {
      min-height: 0;
      padding: 12px 34px 20px;
      display: grid;
      grid-template-rows: var(--rows, 1fr);
      gap: var(--gap, 14px);
      align-items: stretch;
    }
    .row {
      min-height: 0;
      display: grid;
      grid-template-columns: var(--split, minmax(0, 1fr) minmax(0, 1fr));
      gap: var(--gap, 14px);
      align-items: stretch;
    }
    .pane, .stack {
      min-width: 0;
      min-height: 0;
      display: flex;
      flex-direction: column;
      gap: var(--stack-gap, 10px);
    }
    .pane {
      padding: 0;
    }
    .pane.box {
      padding: 12px;
      border: 2px solid var(--mist);
      border-radius: 8px;
      background: #fff;
    }
    .pane.soft { background: var(--soft); }
    .pane.warm { background: var(--warm); }
    .mini-grid {
      display: grid;
      grid-template-columns: var(--split, minmax(0, 1fr) minmax(0, 1fr));
      gap: var(--gap, 12px);
      align-items: stretch;
    }
    .fragment {
      opacity: 0;
      transform: translateY(10px);
      transition: opacity .24s ease, transform .24s ease;
    }
    .fragment.shown {
      opacity: 1;
      transform: translateY(0);
    }
    .label {
      color: var(--teal);
      font-size: clamp(15px, 1.05vw, 20px);
      line-height: 1.12;
      font-weight: 850;
    }
    .eyebrow {
      margin-bottom: 5px;
      color: var(--brown);
      font-size: 11px;
      font-weight: 850;
      letter-spacing: 1.1px;
      text-transform: uppercase;
    }
    .text {
      font-size: clamp(14px, .96vw, 18px);
      line-height: 1.3;
      font-weight: 560;
    }
    .text strong { color: var(--teal); font-weight: 850; }
    .tiny {
      color: #49666b;
      font-size: clamp(11px, .74vw, 14px);
      line-height: 1.25;
      font-style: italic;
    }
    .card, .figure, .code {
      border: 2px solid var(--mist);
      border-radius: 8px;
      background: #fff;
    }
    .card {
      padding: 12px 13px;
    }
    .card.soft { background: var(--soft); }
    .card.warm { background: var(--warm); }
    .card p {
      margin: 0;
      font-size: clamp(13px, .9vw, 17px);
      line-height: 1.32;
      font-weight: 570;
    }
    .figure {
      padding: 9px;
      display: flex;
      flex-direction: column;
      min-height: 0;
    }
    .figure.grow { flex: 1; min-height: 0; }
    .figure img {
      width: 100%;
      min-height: 0;
      object-fit: contain;
      display: block;
    }
    .figure.grow img { flex: 1; }
    .caption {
      margin-top: 5px;
      color: #49666b;
      font-size: clamp(10px, .68vw, 13px);
      line-height: 1.22;
      font-style: italic;
    }
    .diagram {
      min-height: 132px;
      overflow: visible;
    }
    .diagram.tall { min-height: 210px; flex: 1; }
    .diagram.compact {
      min-height: 0;
      height: var(--diagram-h, 132px);
      flex: 0 0 var(--diagram-h, 132px);
    }
    .diagram svg {
      width: 100%;
      height: 100%;
      display: block;
      overflow: visible;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .stat {
      padding: 9px 10px;
      border: 1px solid var(--mist);
      border-radius: 8px;
      background: var(--soft);
    }
    .stat-label {
      color: var(--brown);
      font-size: 11px;
      font-weight: 850;
      letter-spacing: .8px;
    }
    .stat-value {
      margin-top: 3px;
      color: var(--teal);
      font-size: clamp(22px, 1.55vw, 30px);
      line-height: 1;
      font-weight: 900;
    }
    .stat.alarm .stat-value { color: var(--alarm); }
    .code {
      overflow: hidden;
      background: var(--soft);
    }
    .code .tag {
      display: inline-block;
      padding: 5px 10px;
      border-bottom-right-radius: 8px;
      background: var(--gold);
      color: #fff;
      font-size: 12px;
      font-weight: 850;
    }
    pre {
      margin: 0;
      padding: 10px 12px 12px;
      color: var(--ink);
      font-family: var(--mono);
      font-size: clamp(11px, .78vw, 15px);
      line-height: 1.35;
      white-space: pre-wrap;
    }
    .out { color: var(--alarm); font-weight: 750; }
    .comment { color: var(--brown); font-style: italic; }
    .equation {
      padding: 11px 12px;
      border-left: 7px solid var(--gold);
      border-radius: 8px;
      background: var(--warm);
      color: var(--teal);
      font-family: var(--mono);
      font-size: clamp(17px, 1.25vw, 25px);
      line-height: 1.45;
      font-weight: 850;
      text-align: center;
    }
    .alarm { color: var(--alarm); }
    .xbar {
      position: relative;
      display: inline-block;
      line-height: .95;
      vertical-align: baseline;
    }
    .xbar::before {
      content: "";
      position: absolute;
      left: 0;
      right: 0;
      top: -.08em;
      border-top: .08em solid currentColor;
    }
    .flow {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px;
    }
    .flow span {
      padding: 5px 9px;
      border: 1px solid var(--mist);
      border-radius: 999px;
      background: var(--soft);
      font-size: clamp(12px, .82vw, 16px);
      font-weight: 800;
    }
    .flow .arrow {
      padding: 0;
      border: 0;
      background: transparent;
      color: var(--brown);
    }
    .callout {
      padding: 10px 12px;
      border-left: 7px solid var(--gold);
      border-radius: 8px;
      background: var(--soft);
      font-size: clamp(13px, .9vw, 17px);
      line-height: 1.28;
      font-weight: 650;
    }
    #dots {
      display: none;
    }
    .dot {
      width: 10px;
      height: 10px;
      border: 0;
      border-radius: 50%;
      background: #3a5e64;
      pointer-events: auto;
      cursor: pointer;
    }
    .dot.on { background: var(--gold); transform: scale(1.14); }
    #tools {
      position: fixed;
      top: 14px;
      right: 16px;
      z-index: 22;
      display: flex;
      gap: 8px;
      align-items: center;
      color: var(--mist);
      font-size: 13px;
      letter-spacing: .3px;
    }
    #tools button {
      border: 1px solid #4d747a;
      border-radius: 7px;
      padding: 6px 9px;
      background: #153d43;
      color: var(--mist);
      font: inherit;
      cursor: pointer;
    }
    #tools button.on {
      border-color: var(--gold);
      background: var(--gold);
      color: #0d2b30;
    }
    #hint {
      position: fixed;
      bottom: 12px;
      left: 0;
      right: 0;
      z-index: 20;
      text-align: center;
      color: var(--mist);
      font-size: 13px;
      letter-spacing: .5px;
      opacity: .72;
    }
    #counter {
      position: fixed;
      right: 16px;
      bottom: 12px;
      z-index: 20;
      color: var(--mist);
      font-size: 13px;
      letter-spacing: .5px;
      opacity: .85;
    }
    #ink {
      position: absolute;
      inset: 0;
      z-index: 15;
      width: 100%;
      height: 100%;
      pointer-events: none;
      touch-action: none;
    }
    #deck.drawing #ink {
      pointer-events: auto;
      cursor: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='22' height='22' viewBox='0 0 28 28'><path d='M4 24l1.2-4.4L18.9 5.9a2.5 2.5 0 0 1 3.5 3.5L8.7 23.1 4 24z' fill='%23e07a2f' stroke='%23073b42' stroke-width='1.4' stroke-linejoin='round'/><path d='M17.4 7.4l3.5 3.5' stroke='%23073b42' stroke-width='1.4'/></svg>") 3 19, crosshair;
    }
    @media (max-width: 900px) {
      #stage { padding: 34px 8px 34px; }
      #deck { width: 98vw; height: 90vh; aspect-ratio: auto; }
      .slide { overflow: hidden; }
      .slide-body { grid-template-rows: none; overflow-y: auto; }
      .row, .mini-grid { grid-template-columns: 1fr !important; }
      .slide-head { padding-inline: 24px; }
    }
  </style>
</head>
<body>
  <div id="dots"></div>
  <div id="tools">
    <button id="pencil" type="button">Pencil: off</button>
    <button id="undo" type="button">Undo</button>
    <button id="clear" type="button">Clear</button>
  </div>

  <div id="stage">
    <main id="deck" aria-live="polite">
      <section class="slide" data-slide>
        <div class="chrome-top"></div>
        <header class="slide-head">
          <div class="kicker">Donut audit</div>
          <h1>Bob's Donut Cafe | Introduction to the one-sample t-test</h1>
          <div class="title-line"></div>
        </header>
        <div class="slide-body" style="--rows: 1fr; --gap:14px">
          <div class="row" style="--split: .92fr 1.58fr; --gap:18px">
            <div class="pane" style="--stack-gap:12px">
              <div class="label fragment">Their claim</div>
              <div class="diagram tall fragment">
                <svg viewBox="0 0 360 360">
                  <!-- cafe name board -->
                  <rect x="96" y="20" width="168" height="34" rx="7" fill="#073b42"/>
                  <text x="180" y="44" text-anchor="middle" font-family="Inter, Arial" font-size="17" font-weight="900" fill="#d4af33">Bob's Donut Cafe</text>
                  <!-- awning -->
                  <path d="M44 96 H316 L292 58 H68 Z" fill="#035f6c"/>
                  <path d="M68 58 H292 L274 30 H86 Z" fill="#d4af33"/>
                  <!-- building -->
                  <rect x="40" y="96" width="280" height="150" rx="8" fill="#f2f7f7" stroke="#c4d8db" stroke-width="4"/>
                  <!-- claim sign -->
                  <rect x="74" y="108" width="212" height="50" rx="7" fill="#fbeecb" stroke="#d4af33" stroke-width="3"/>
                  <text x="180" y="122" text-anchor="middle" font-family="Inter, Arial" font-size="11" font-weight="900" fill="#7a6212">BOB'S SIGNATURE DONUT</text>
                  <text x="180" y="143" text-anchor="middle" font-family="Inter, Arial" font-size="19" font-weight="950" fill="#035f6c">only 10g sugar</text>
                  <text x="180" y="156" text-anchor="middle" font-family="Inter, Arial" font-size="10" font-weight="850" fill="#7a6212">on average</text>
                  <!-- windows -->
                  <rect x="56" y="172" width="74" height="62" fill="#ffffff" stroke="#c4d8db" stroke-width="4"/>
                  <rect x="230" y="172" width="74" height="62" fill="#ffffff" stroke="#c4d8db" stroke-width="4"/>
                  <!-- door -->
                  <rect x="150" y="180" width="60" height="66" fill="#e7c9a9" stroke="#caa884" stroke-width="4"/>
                  <circle cx="200" cy="214" r="4" fill="#7a6212"/>
                  <!-- donuts in windows -->
                  <circle cx="93" cy="203" r="17" fill="#d4af33" stroke="#7a6212" stroke-width="3"/>
                  <circle cx="93" cy="203" r="7" fill="#fbeecb"/>
                  <circle cx="267" cy="203" r="17" fill="#d4af33" stroke="#7a6212" stroke-width="3"/>
                  <circle cx="267" cy="203" r="7" fill="#fbeecb"/>
                  <!-- stick figure (you) -->
                  <circle cx="64" cy="278" r="14" fill="#e7c9a9" stroke="#caa884" stroke-width="3"/>
                  <line x1="64" y1="292" x2="64" y2="324" stroke="#035f6c" stroke-width="5" stroke-linecap="round"/>
                  <line x1="64" y1="302" x2="48" y2="316" stroke="#035f6c" stroke-width="5" stroke-linecap="round"/>
                  <line x1="64" y1="302" x2="80" y2="316" stroke="#035f6c" stroke-width="5" stroke-linecap="round"/>
                  <line x1="64" y1="324" x2="50" y2="344" stroke="#035f6c" stroke-width="5" stroke-linecap="round"/>
                  <line x1="64" y1="324" x2="78" y2="344" stroke="#035f6c" stroke-width="5" stroke-linecap="round"/>
                  <text x="64" y="357" text-anchor="middle" font-family="Inter, Arial" font-size="11" font-weight="800" fill="#49666b">you (competitor)</text>
                  <!-- thought bubble -->
                  <circle cx="92" cy="288" r="5" fill="#fff" stroke="#c4d8db" stroke-width="2"/>
                  <circle cx="104" cy="278" r="7" fill="#fff" stroke="#c4d8db" stroke-width="2"/>
                  <rect x="104" y="256" width="246" height="58" rx="22" fill="#ffffff" stroke="#c4d8db" stroke-width="2.5"/>
                  <text x="227" y="282" text-anchor="middle" font-family="Inter, Arial" font-size="15" font-weight="700" font-style="italic" fill="#49666b">Looks great...</text>
                  <text x="227" y="304" text-anchor="middle" font-family="Inter, Arial" font-size="16" font-weight="900" fill="#b5451f">but really 10g?</text>
                </svg>
              </div>
              <div class="text fragment">Bob's Donut Cafe claims: <strong>On average, Bob's Signature Donut contains only 10g sugar.</strong></div>
              <div class="tiny fragment">Some donut-to-donut variation is expected. The test is about the long-run mean.</div>
            </div>

            <div class="pane" style="--stack-gap:14px">
              <div class="row" style="--split: .88fr 1.12fr; --gap:14px">
                <div class="pane">
                  <div class="label fragment">Audit plan</div>
                  <div class="text fragment">Buy one donut on each of ten random days.</div>
                  <div class="text fragment">Lab-test sugar the same way each time.</div>
                  <div class="code fragment">
                    <div class="tag">R</div>
                    <pre>sugar &lt;- c(9.8, 14.6, 8.8, 11.8, 15.4,
          10.3, 10.6, 10.2, 11.0, 11.9)</pre>
                  </div>
                </div>
                <div class="pane">
                  <div class="label fragment">Observed sample</div>
                  <div class="figure grow fragment">
                    <img src="figures/fig1_dots.png" alt="">
                    <div class="caption">Dough-colored dots: donuts. Gold line: claim. Red dashed line: sample mean.</div>
                  </div>
                </div>
              </div>

              <div class="row" style="--split: .74fr 1.26fr; --gap:14px">
                <div class="pane">
                  <div class="label fragment">Question</div>
                  <div class="text fragment">The sample average is higher than the claim.</div>
                  <div class="text fragment">Observed donuts ranged from 8.8g to 15.4g sugar.</div>
                  <div class="callout fragment">Is the gap large relative to ordinary sampling wobble?</div>
                </div>
                <div class="pane">
                  <div class="label fragment">Minimal ggplot2 code</div>
                  <div class="code fragment">
                    <div class="tag">ggplot2</div>
                    <pre>donut_dough &lt;- "#c98245"
donut_edge &lt;- "#7a4a22"

ggplot(tibble(sugar), aes(sugar)) +
  geom_dotplot(binwidth = .5, fill = donut_dough,
               colour = donut_edge) +
  geom_vline(xintercept = 10, colour = gold) +
  geom_vline(xintercept = mean(sugar), colour = alarm,
             linetype = "dashed")</pre>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div class="chrome-bottom"></div>
      </section>

      <section class="slide" data-slide>
        <div class="chrome-top"></div>
        <header class="slide-head">
          <div class="kicker">Build the null model</div>
          <h1>To draw the null bell curve, we need a center and a spread</h1>
          <div class="title-line"></div>
        </header>
        <div class="slide-body" style="--rows: auto 1fr; --gap:14px">
          <div class="row" style="--split: .9fr 1.1fr; --gap:18px">
            <div class="pane">
              <div class="label fragment">Known center</div>
              <div class="text fragment">The null gives us the center.</div>
              <div class="equation fragment">H<sub>0</sub>: &mu; = 10</div>
              <div class="text fragment">If the claim is true, repeated sample means should cluster around 10g.</div>
            </div>
            <div class="pane">
              <div class="label fragment">Unknown donut spread</div>
              <div class="mini-grid" style="--split: .92fr 1.08fr; --gap:14px">
                <div class="text fragment">We do not know the population standard deviation, &sigma;.</div>
                <div class="diagram compact fragment" style="--diagram-h:155px" aria-hidden="true">
                  <svg viewBox="0 0 340 210">
                    <text x="24" y="28" font-family="Inter, Arial" font-size="15" font-weight="850" fill="#7a6212">ALL DONUTS</text>
                    <g opacity=".9">
                      <circle cx="55" cy="88" r="16" fill="none" stroke="#035f6c" stroke-width="5"/>
                      <circle cx="55" cy="88" r="6" fill="#fbeecb"/>
                      <circle cx="102" cy="62" r="16" fill="none" stroke="#b5451f" stroke-width="5"/>
                      <circle cx="102" cy="62" r="6" fill="#fbeecb"/>
                      <circle cx="149" cy="102" r="16" fill="none" stroke="#035f6c" stroke-width="5"/>
                      <circle cx="149" cy="102" r="6" fill="#fbeecb"/>
                      <circle cx="196" cy="72" r="16" fill="none" stroke="#035f6c" stroke-width="5"/>
                      <circle cx="196" cy="72" r="6" fill="#fbeecb"/>
                      <circle cx="243" cy="110" r="16" fill="none" stroke="#d4af33" stroke-width="5"/>
                      <circle cx="243" cy="110" r="6" fill="#fbeecb"/>
                      <circle cx="290" cy="82" r="16" fill="none" stroke="#035f6c" stroke-width="5"/>
                      <circle cx="290" cy="82" r="6" fill="#fbeecb"/>
                    </g>
                    <text x="170" y="160" text-anchor="middle" font-family="Inter, Arial" font-size="21" font-weight="900" fill="#073b42">How wide is this?</text>
                    <text x="170" y="188" text-anchor="middle" font-family="Inter, Arial" font-size="17" font-weight="700" fill="#49666b">We estimate it from the audit.</text>
                  </svg>
                </div>
              </div>
            </div>
          </div>

          <div class="row" style="--split: .92fr .78fr 1fr; --gap:14px">
            <div class="pane">
              <div class="label fragment">Estimate spread</div>
              <div class="text fragment">Use the sample standard deviation as the best available estimate.</div>
              <div class="equation fragment">s = 2.09</div>
              <div class="figure grow fragment">
                <img src="figures/fig2_null.png" alt="">
                <div class="caption">Center at 10g. Width borrowed from the audit.</div>
              </div>
            </div>

            <div class="pane">
              <div class="label fragment">Mean spread</div>
              <div class="text fragment">The test is about a mean, so use the spread of sample means.</div>
              <div class="equation fragment">SE = s / sqrt(n)<br><span class="alarm">SE = 0.66</span></div>
              <div class="diagram compact fragment" style="--diagram-h:185px">
                <svg viewBox="0 0 340 210">
                  <g>
                    <circle cx="34" cy="68" r="10" fill="#035f6c"/>
                    <circle cx="64" cy="68" r="10" fill="#b5451f"/>
                    <circle cx="94" cy="87" r="10" fill="#035f6c"/>
                    <circle cx="124" cy="87" r="10" fill="#035f6c"/>
                    <circle cx="154" cy="87" r="10" fill="#b5451f"/>
                    <circle cx="184" cy="87" r="10" fill="#035f6c"/>
                    <circle cx="214" cy="87" r="10" fill="#035f6c"/>
                    <circle cx="244" cy="76" r="10" fill="#035f6c"/>
                    <circle cx="274" cy="68" r="10" fill="#035f6c"/>
                    <circle cx="304" cy="68" r="10" fill="#035f6c"/>
                  </g>
                  <path d="M58 120 C130 154 235 154 286 120" fill="none" stroke="#d4af33" stroke-width="5" stroke-linecap="round"/>
                  <path d="M170 122 V154" stroke="#d4af33" stroke-width="5" stroke-linecap="round"/>
                  <path d="M170 164 L158 148 H182 Z" fill="#d4af33"/>
                  <rect x="74" y="164" width="192" height="36" rx="8" fill="#035f6c"/>
                  <text x="170" y="188" text-anchor="middle" font-family="Inter, Arial" font-size="17" font-weight="850" fill="#fff">sample mean</text>
                </svg>
              </div>
            </div>

            <div class="pane">
              <div class="label fragment">Null curve for <span class="xbar">x</span></div>
              <div class="figure grow fragment">
                <img src="figures/fig3_se.png" alt="">
                <div class="caption">Same null center, but the mean has narrower wobble.</div>
              </div>
              <div class="callout fragment">Null bell curve for <span class="xbar">x</span>: center 10g, spread SE = 0.66.</div>
            </div>
          </div>
        </div>
        <div class="chrome-bottom"></div>
      </section>

      <section class="slide" data-slide>
        <div class="chrome-top"></div>
        <header class="slide-head">
          <div class="kicker">Measure surprise</div>
          <h1>The t statistic says how far the audit mean sits from 10g</h1>
          <div class="title-line"></div>
        </header>
        <div class="slide-body" style="--rows: auto 1fr; --gap:14px">
          <div class="row" style="--split: .86fr 1.14fr; --gap:18px">
            <div class="pane">
              <div class="label fragment">Observed gap</div>
              <div class="text fragment"><span class="xbar">x</span> - &mu;<sub>0</sub> = 11.44 - 10 = 1.44.</div>
              <div class="code fragment">
                <div class="tag">R</div>
                <pre>mean(sugar) - 10
<span class="out">#&gt; 1.44</span>
sd(sugar) / sqrt(length(sugar))
<span class="out">#&gt; 0.66</span></pre>
              </div>
            </div>

            <div class="pane">
              <div class="label fragment">Standardize</div>
              <div class="equation fragment">t = (<span class="xbar">x</span> - &mu;<sub>0</sub>) / SE<br><span class="alarm">t = 1.44 / 0.66 = 2.18</span></div>
              <div class="diagram compact fragment" style="--diagram-h:165px">
                <svg viewBox="0 0 420 220">
                  <line x1="44" y1="132" x2="376" y2="132" stroke="#c4d8db" stroke-width="8" stroke-linecap="round"/>
                  <g font-family="Inter, Arial" font-size="16" font-weight="850" fill="#073b42" text-anchor="middle">
                    <line x1="44" y1="104" x2="44" y2="158" stroke="#035f6c" stroke-width="3"/><text x="44" y="192">-3</text>
                    <line x1="99" y1="104" x2="99" y2="158" stroke="#035f6c" stroke-width="3"/><text x="99" y="192">-2</text>
                    <line x1="155" y1="104" x2="155" y2="158" stroke="#035f6c" stroke-width="3"/><text x="155" y="192">-1</text>
                    <line x1="210" y1="104" x2="210" y2="158" stroke="#035f6c" stroke-width="3"/><text x="210" y="192">0</text>
                    <line x1="265" y1="104" x2="265" y2="158" stroke="#035f6c" stroke-width="3"/><text x="265" y="192">1</text>
                    <line x1="321" y1="104" x2="321" y2="158" stroke="#035f6c" stroke-width="3"/><text x="321" y="192">2</text>
                    <line x1="376" y1="104" x2="376" y2="158" stroke="#035f6c" stroke-width="3"/><text x="376" y="192">3</text>
                  </g>
                  <line x1="331" y1="58" x2="331" y2="154" stroke="#b5451f" stroke-width="7" stroke-linecap="round"/>
                  <circle cx="331" cy="58" r="17" fill="#b5451f"/>
                  <text x="331" y="33" text-anchor="middle" font-family="Inter, Arial" font-size="21" font-weight="900" fill="#b5451f">t = 2.18</text>
                </svg>
              </div>
            </div>
          </div>

          <div class="row" style="--split: 1.18fr .82fr; --gap:18px">
            <div class="pane">
              <div class="label fragment">Reference curve</div>
              <div class="figure grow fragment">
                <img src="figures/fig4_tdist.png" alt="">
                <div class="caption">df = 9 because s was estimated from 10 donuts.</div>
              </div>
            </div>

            <div class="pane">
              <div class="label fragment">Decision</div>
              <div class="text fragment">Two-sided p = 0.0575.</div>
              <div class="card warm fragment">
                <p>At alpha = 0.05, the evidence is suggestive but not quite enough to reject H0.</p>
              </div>
              <div class="code fragment">
                <div class="tag">R</div>
                <pre>t.test(sugar, mu = 10)
<span class="out">#&gt; t = 2.18, df = 9</span>
<span class="out">#&gt; p = 0.0575</span></pre>
              </div>
            </div>
          </div>
        </div>
        <div class="chrome-bottom"></div>
      </section>

      <section class="slide" data-slide>
        <div class="chrome-top"></div>
        <header class="slide-head">
          <div class="kicker">Precision changes the ending</div>
          <h1>More audit days shrink SE, so the same signal becomes clearer</h1>
          <div class="title-line"></div>
        </header>
        <div class="slide-body" style="--rows: auto 1fr; --gap:14px">
          <div class="row" style="--split: 1fr 1fr 1fr; --gap:14px">
            <div class="pane">
              <div class="label fragment">n = 10</div>
              <div class="text fragment"><span class="xbar">x</span> = 11.44, SE = 0.66, p = 0.0575.</div>
              <div class="card soft fragment"><p>Higher than the claim, but not enough for a two-sided 0.05 rejection.</p></div>
            </div>
            <div class="pane">
              <div class="label fragment">n = 30</div>
              <div class="text fragment"><span class="xbar">x</span> = 11.34, SE = 0.44, p = 0.0045.</div>
              <div class="card warm fragment"><p>The mean barely moved. The precision changed.</p></div>
            </div>
            <div class="pane">
              <div class="label fragment">Reporting</div>
              <div class="code fragment">
                <div class="tag">R</div>
                <pre>t.test(sugar, mu = 10)
<span class="out">#&gt; p = 0.0575</span>
 t.test(sugar30, mu = 10)
<span class="out">#&gt; p = 0.0045</span></pre>
              </div>
            </div>
          </div>

          <div class="row" style="--split: 1.1fr .9fr; --gap:18px">
            <div class="pane">
              <div class="label fragment">Intervals</div>
              <div class="figure grow fragment">
                <img src="figures/fig5_intervals.png" alt="">
                <div class="caption">n = 10 touches 10g. n = 30 clears 10g.</div>
              </div>
            </div>
            <div class="pane">
              <div class="label fragment">Power</div>
              <div class="text fragment">Power is the probability of rejecting H0 when a specified alternative is true.</div>
              <div class="callout fragment">Larger n -> smaller SE -> higher power for the same true gap.</div>
              <div class="diagram fragment">
                <svg viewBox="0 0 340 220">
                  <g fill="none" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M36 158 C70 154 85 124 112 112 C141 99 165 149 195 153 C226 158 259 158 304 158" stroke="#c4d8db" stroke-width="18" opacity=".7"/>
                    <path d="M46 158 C82 158 100 91 128 91 C156 91 174 158 210 158" stroke="#d4af33" stroke-width="12" opacity=".86"/>
                    <line x1="70" y1="52" x2="70" y2="176" stroke="#035f6c" stroke-width="5"/>
                    <line x1="156" y1="52" x2="156" y2="176" stroke="#b5451f" stroke-width="5"/>
                  </g>
                  <g font-family="Inter, Arial" font-weight="900" fill="#073b42" text-anchor="middle">
                    <text x="70" y="38" font-size="16">claim</text>
                    <text x="156" y="38" font-size="16">same gap</text>
                    <text x="169" y="200" font-size="17" fill="#035f6c">wide SE -> borderline</text>
                    <text x="169" y="177" font-size="17" fill="#7a6212">narrow SE -> clearer</text>
                  </g>
                </svg>
              </div>
              <div class="text fragment">Say what changed: precision, not the basic pattern.</div>
            </div>
          </div>
        </div>
        <div class="chrome-bottom"></div>
      </section>

      <section class="slide" data-slide>
        <div class="chrome-top"></div>
        <header class="slide-head">
          <div class="kicker">Report the story</div>
          <h1>Use decision language that matches the evidence</h1>
          <div class="title-line"></div>
        </header>
        <div class="slide-body" style="--rows: auto 1fr; --gap:14px">
          <div class="row" style="--split: .95fr 1.05fr; --gap:18px">
            <div class="pane">
              <div class="label fragment">n = 10 report</div>
              <div class="card soft fragment">
                <p>The audit mean was 1.44g above the claim, but the two-sided test did not reach alpha = 0.05.</p>
              </div>
              <div class="text fragment">Do not say "no difference." Say "insufficient evidence with n = 10."</div>
            </div>
            <div class="pane">
              <div class="label fragment">One-sided option</div>
              <div class="mini-grid" style="--split: .9fr 1.1fr">
                <div class="card warm fragment">
                  <p>H1: &mu; &gt; 10 is valid only if that direction was chosen before seeing the data.</p>
                </div>
                <div class="code fragment">
                  <div class="tag">R</div>
                  <pre>t.test(sugar, mu = 10,
       alternative = "greater")
<span class="out">#&gt; one-sided p = 0.0288</span>
<span class="comment"># direction chosen before data</span></pre>
                </div>
              </div>
            </div>
          </div>

          <div class="row" style="--split: 1.28fr .72fr; --gap:18px">
            <div class="pane">
              <div class="label fragment">Full chain</div>
              <div class="flow fragment">
                <span>claim</span><span class="arrow">-&gt;</span>
                <span>H0</span><span class="arrow">-&gt;</span>
                <span>gap</span><span class="arrow">-&gt;</span>
                <span>s</span><span class="arrow">-&gt;</span>
                <span>SE</span><span class="arrow">-&gt;</span>
                <span>t</span><span class="arrow">-&gt;</span>
                <span>p</span><span class="arrow">-&gt;</span>
                <span>decision</span>
              </div>
              <div class="callout fragment">The bell curve under H0 is not magic: center from H0, spread estimated from data, then shrink by sqrt(n).</div>
              <div class="diagram fragment">
                <svg viewBox="0 0 410 170">
                  <g font-family="Inter, Arial" font-weight="900" text-anchor="middle">
                    <rect x="18" y="30" width="92" height="48" rx="10" fill="#fbeecb" stroke="#d4af33" stroke-width="3"/>
                    <text x="64" y="60" font-size="17" fill="#035f6c">10g</text>
                    <rect x="158" y="30" width="92" height="48" rx="10" fill="#f2f7f7" stroke="#c4d8db" stroke-width="3"/>
                    <text x="204" y="60" font-size="17" fill="#035f6c">s = 2.09</text>
                    <rect x="300" y="30" width="92" height="48" rx="10" fill="#f2f7f7" stroke="#c4d8db" stroke-width="3"/>
                    <text x="346" y="60" font-size="17" fill="#035f6c">SE = 0.66</text>
                    <path d="M112 54 H150" stroke="#7a6212" stroke-width="3"/>
                    <path d="M252 54 H292" stroke="#7a6212" stroke-width="3"/>
                    <text x="134" y="41" font-size="17" fill="#7a6212">+</text>
                    <text x="272" y="41" font-size="17" fill="#7a6212">/ sqrt(n)</text>
                  </g>
                  <path d="M54 132 C90 132 100 95 128 95 C156 95 166 132 202 132" fill="none" stroke="#c4d8db" stroke-width="13" stroke-linecap="round"/>
                  <path d="M212 132 C244 132 256 78 278 78 C300 78 312 132 344 132" fill="none" stroke="#d4af33" stroke-width="12" stroke-linecap="round"/>
                  <text x="204" y="160" text-anchor="middle" font-family="Inter, Arial" font-size="16" font-weight="900" fill="#073b42">same center, narrower curve for the mean</text>
                </svg>
              </div>
            </div>
            <div class="pane">
              <div class="label fragment">Takeaway</div>
              <div class="text fragment">A one-sample t-test measures a sample-mean gap against the wobble expected when the claim is true.</div>
              <div class="text fragment">The story is about uncertainty, not just a p-value.</div>
            </div>
          </div>
        </div>
        <div class="chrome-bottom"></div>
      </section>

      <canvas id="ink"></canvas>
    </main>
  </div>
  <div id="hint">Left/right reveal | up/down slides | P pencil | U undo | C clear | [ ] pen | R reset</div>
  <div id="counter"></div>

  <script>
    const slides = [...document.querySelectorAll('.slide')];
    const dots = document.getElementById('dots');
    const deck = document.getElementById('deck');
    const counter = document.getElementById('counter');
    const pencilButton = document.getElementById('pencil');
    const undoButton = document.getElementById('undo');
    const clearButton = document.getElementById('clear');
    const canvas = document.getElementById('ink');
    const ctx = canvas.getContext('2d');
    const WIDTH = 1920;
    const HEIGHT = 1080;
    let slideIndex = 0;
    let shown = 0;
    let fragments = [];
    let pencilMode = false;
    let penWidth = 4;
    let activeStroke = null;
    const drawings = slides.map(() => []);

    canvas.width = WIDTH;
    canvas.height = HEIGHT;

    slides.forEach((_, index) => {
      const dot = document.createElement('button');
      dot.type = 'button';
      dot.className = 'dot';
      dot.title = `Slide ${index + 1}`;
      dot.addEventListener('click', event => {
        event.stopPropagation();
        showSlide(index);
      });
      dots.appendChild(dot);
    });

    function showSlide(index, revealAll = false) {
      slideIndex = (index + slides.length) % slides.length;
      shown = 0;
      slides.forEach((slide, i) => slide.classList.toggle('active', i === slideIndex));
      fragments = [...slides[slideIndex].querySelectorAll('.fragment')];
      if (revealAll) shown = fragments.length;
      render();
      drawStrokes();
    }

    function render() {
      fragments.forEach((fragment, index) => {
        fragment.classList.toggle('shown', index < shown);
      });
      [...dots.children].forEach((dot, index) => dot.classList.toggle('on', index === slideIndex));
      counter.textContent = `${slideIndex + 1} / ${slides.length} | ${shown} / ${fragments.length} | pen ${penWidth}px`;
    }

    function nextStep() {
      if (shown < fragments.length) {
        shown += 1;
        render();
      } else {
        showSlide(slideIndex + 1);
      }
    }

    function prevStep() {
      if (shown > 0) {
        shown -= 1;
        render();
      } else {
        showSlide(slideIndex - 1, true);
      }
    }

    function resetSlide() {
      shown = 0;
      render();
    }

    function setPencil(on) {
      pencilMode = on;
      deck.classList.toggle('drawing', pencilMode);
      pencilButton.classList.toggle('on', pencilMode);
      pencilButton.textContent = `Pencil: ${pencilMode ? 'on' : 'off'}`;
    }

    function pointFor(event) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: (event.clientX - rect.left) * WIDTH / rect.width,
        y: (event.clientY - rect.top) * HEIGHT / rect.height
      };
    }

    function drawOne(stroke) {
      if (stroke.points.length < 2) return;
      ctx.save();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.strokeStyle = stroke.color;
      ctx.lineWidth = stroke.width;
      ctx.beginPath();
      ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
      stroke.points.slice(1).forEach(pt => ctx.lineTo(pt.x, pt.y));
      ctx.stroke();
      ctx.restore();
    }

    function drawStrokes() {
      ctx.clearRect(0, 0, WIDTH, HEIGHT);
      drawings[slideIndex].forEach(drawOne);
    }

    canvas.addEventListener('click', event => event.stopPropagation());
    canvas.addEventListener('pointerdown', event => {
      if (!pencilMode) return;
      event.preventDefault();
      event.stopPropagation();
      canvas.setPointerCapture(event.pointerId);
      activeStroke = { color: '#e07a2f', width: penWidth, points: [pointFor(event)] };
      drawings[slideIndex].push(activeStroke);
    });
    canvas.addEventListener('pointermove', event => {
      if (!activeStroke || !pencilMode) return;
      event.preventDefault();
      event.stopPropagation();
      activeStroke.points.push(pointFor(event));
      drawStrokes();
    });
    function finishStroke(event) {
      if (!activeStroke) return;
      event.preventDefault();
      event.stopPropagation();
      activeStroke = null;
      drawStrokes();
    }
    canvas.addEventListener('pointerup', finishStroke);
    canvas.addEventListener('pointercancel', finishStroke);
    canvas.addEventListener('lostpointercapture', () => { activeStroke = null; });

    function undoStroke() {
      drawings[slideIndex].pop();
      drawStrokes();
    }

    function clearStrokes() {
      drawings[slideIndex] = [];
      drawStrokes();
    }

    pencilButton.addEventListener('click', event => {
      event.stopPropagation();
      setPencil(!pencilMode);
    });
    undoButton.addEventListener('click', event => {
      event.stopPropagation();
      undoStroke();
    });
    clearButton.addEventListener('click', event => {
      event.stopPropagation();
      clearStrokes();
    });
    window.addEventListener('resize', drawStrokes);

    document.addEventListener('keydown', event => {
      if (['ArrowRight', ' ', 'PageDown', 'Enter'].includes(event.key)) {
        event.preventDefault();
        nextStep();
      } else if (['ArrowLeft', 'PageUp'].includes(event.key)) {
        event.preventDefault();
        prevStep();
      } else if (event.key === 'ArrowDown') {
        event.preventDefault();
        showSlide(slideIndex + 1);
      } else if (event.key === 'ArrowUp') {
        event.preventDefault();
        showSlide(slideIndex - 1);
      } else if (event.key.toLowerCase() === 'p') {
        event.preventDefault();
        setPencil(!pencilMode);
      } else if (event.key.toLowerCase() === 'u') {
        event.preventDefault();
        undoStroke();
      } else if (event.key.toLowerCase() === 'c') {
        event.preventDefault();
        clearStrokes();
      } else if (event.key === '[') {
        event.preventDefault();
        penWidth = Math.max(1, penWidth - 1);
        render();
      } else if (event.key === ']') {
        event.preventDefault();
        penWidth = Math.min(18, penWidth + 1);
        render();
      } else if (event.key.toLowerCase() === 'r') {
        event.preventDefault();
        resetSlide();
      }
    });

    deck.addEventListener('click', () => {
      if (!pencilMode) nextStep();
    });

    const params = new URLSearchParams(window.location.search);
    const initialSlide = Number.parseInt(params.get('slide') || '0', 10);
    const revealAll = params.get('all') === '1';
    showSlide(Number.isFinite(initialSlide) ? initialSlide : 0, revealAll);
  </script>
</body>
</html>
'''


def main():
    for svg in HERE.glob("slide-*.svg"):
        svg.unlink()
    (HERE / "index.html").write_text(HTML, encoding="utf-8")


if __name__ == "__main__":
    main()
