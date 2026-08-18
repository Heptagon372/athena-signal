"use client";

/**
 * 올빼미 — 확신도·방향·장 세션에 따라 표정이 달라집니다.
 *   강한 상승  날개 활짝, 귀깃 쫑긋, 눈썹 올라감, 동공 확대
 *   불확실     동공 축소, 눈썹 수평
 *   강한 하락  날개 접고 웅크림, 미간 찌푸림
 *   장 마감    눈 감고 수면, zzz
 */

import { useEffect, useState } from "react";

const MOODS = {
  sleeping: ["장이 닫혀 쉬는 중", "다음 개장을 기다리는 중"],
  strongUp: ["날개를 활짝 펴고 있어요", "확신에 차 있어요"],
  mildUp: ["조심스럽게 기대하는 중", "고개를 들고 지켜보는 중"],
  neutral: ["어느 쪽도 확신하지 못하는 중", "눈만 껌뻑이며 관망 중"],
  mildDown: ["미간을 좁히고 있어요", "경계하는 눈빛이에요"],
  strongDown: ["날개를 접고 웅크렸어요", "잔뜩 찌푸리고 있어요"],
};

export default function Owl({ direction = "up", probability = 50, marketOpen = true, name = "" }) {
  const [blink, setBlink] = useState(false);

  const up = direction === "up";
  const strong = probability >= 65;
  const weak = probability < 55;

  useEffect(() => {
    if (!marketOpen) return;
    let timer;
    const loop = () => {
      setBlink(true);
      setTimeout(() => setBlink(false), 150);
      timer = setTimeout(loop, 2600 + Math.random() * 4200);
    };
    timer = setTimeout(loop, 1200 + Math.random() * 2000);
    return () => clearTimeout(timer);
  }, [marketOpen]);

  const cls = [
    "owl",
    marketOpen ? (up ? "rising" : "falling") : "sleeping",
    marketOpen && strong ? "strong" : "",
    blink ? "blinking" : "",
  ].filter(Boolean).join(" ");

  const pupilR = weak ? 3.6 : strong ? 6.4 : 5.0;
  const pupilY = 74 + (marketOpen ? (up ? -2.2 : 2.6) : 0);

  const moodKey = !marketOpen ? "sleeping"
    : weak ? "neutral"
    : up ? (strong ? "strongUp" : "mildUp")
         : (strong ? "strongDown" : "mildDown");
  const list = MOODS[moodKey];
  const mood = list[(name.length + Math.round(probability)) % list.length];

  return (
    <div className="owl-stage">
      <svg className={cls} width="200" height="205" viewBox="0 0 200 210">
        <g className="owl-feet">
          <path d="M88,182 l0,9 M84,191 l4,-2 4,2" />
          <path d="M112,182 l0,9 M108,191 l4,-2 4,2" />
        </g>
        <path className="owl-wing wing-l" d="M64,118 Q26,96 18,146 Q46,160 70,138 Z" />
        <path className="owl-wing wing-r" d="M136,118 Q174,96 182,146 Q154,160 130,138 Z" />
        <ellipse className="owl-body" cx="100" cy="130" rx="39" ry="50" />
        <g className="owl-plume">
          <path d="M86,116 q6,7 0,14 M100,112 q6,7 0,14 M114,116 q6,7 0,14" />
          <path d="M79,138 q6,7 0,14 M93,134 q6,7 0,14 M107,134 q6,7 0,14 M121,138 q6,7 0,14" />
          <path d="M86,158 q6,7 0,14 M100,155 q6,7 0,14 M114,158 q6,7 0,14" />
        </g>
        <path className="owl-tuft tuft-l" d="M74,46 L64,16 L89,43 Z" />
        <path className="owl-tuft tuft-r" d="M126,46 L136,16 L111,43 Z" />
        <circle className="owl-head" cx="100" cy="76" r="37" />
        <path className="owl-disc" d="M100,44 Q76,44 72,74 Q70,100 100,104 Q130,100 128,74 Q124,44 100,44 Z" />

        {[84, 116].map((cx, i) => (
          <g key={cx}>
            <circle className="eye-white" cx={cx} cy="74" r="12.5" />
            <circle className="eye-ring" cx={cx} cy="74" r="12.5" />
            <circle className="eye-pupil" cx={cx} cy={pupilY} r={pupilR} />
            <circle className="eye-shine" cx={cx + 3} cy="70.5" r="1.7" />
            <path className="eye-lid"
                  d={`M${cx - 12.5},74 A12.5,12.5 0 0,1 ${cx + 12.5},74 Z`} />
          </g>
        ))}

        <path className="owl-brow brow-l" d="M72,58 L96,58" />
        <path className="owl-brow brow-r" d="M104,58 L128,58" />
        <path className="owl-beak" d="M94,86 L106,86 L100,98 Z" />
        <g className="owl-zzz">
          <text x="146" y="46" fontSize="13">z</text>
          <text x="156" y="34" fontSize="10">z</text>
        </g>
      </svg>

      <div className="owl-caption">
        <b>{name}</b>{" "}
        {marketOpen
          ? `${up ? "상승" : "하락"} 신호, 확신도 ${probability}%`
          : `개장 시 ${up ? "상승" : "하락"} 예상, 확신도 ${probability}%`}
      </div>
      <div className="owl-mood">{mood}</div>
    </div>
  );
}
