raise SystemExit("Historical one-off patch script; rerunning is disabled.")
from pathlib import Path
p=Path('diag/capacity.py'); s=p.read_text()
s=s.replace('GRIDS = {"V1′": (12, 16), "V3": (24, 32), "V2′": (6, 8)}', '''GRIDS = {"V1'": (12, 16), "V3": (24, 32), "V2'": (6, 8)}''')
s=s.replace('                prediction, _, _ = fitter.fit(py, pv, pi)', '                frame_start = time.monotonic()\n                prediction, _, _ = fitter.fit(py, pv, pi)').replace('                score_start = time.monotonic()\n','').replace('(time.monotonic()-score_start)/len(k)', '(time.monotonic()-frame_start)/len(k)')
s=s.replace('"leave_one_F_out": ablations', '"leave_one_coil_out": ablations')
s=s.replace('    return "\\n".join(lines)+"\\n"\n\n\ndef coil_sums', '''    verification_path = HERE/"verification.json"
    if verification_path.is_file():
        verification = json.loads(verification_path.read_text())
        lines += ["", f"合成データ試験: {verification['tests_run']} 件。合格: {verification['successful']}。独立レビューは行わず、自己点検と実行試験で確認しました。"]
    calibration = s.get("calibration")
    if calibration:
        lines += ["", "## 校正の切り分け", "", "|除外するコイル|α_F|β_E|中心化残差平方和|", "|---|---:|---:|---:|"]
        for name, result in calibration["leave_one_coil_out"].items():
            lines.append(f"|{name}|{result['alpha_F']:.6g}|{result['beta_E']:.6g}|{result['centered_sse']:.6g}|")
        straight = calibration["no_shear"]
        lines += ["", f"傾斜なし: α_F={straight['alpha_F']:.6g}、β_E={straight['beta_E']:.6g}、残差平方和={straight['centered_sse']:.6g}。", "",
                  "コイル別の自由係数と固定倍率候補（±1、±1/(2π)、±0.001、±1000）の残差は JSON に保存します。β_E は α_F と別の直接係数であり、有効巻数の解釈には β_E/α_F を使います。"]
    lines += ["", "修正時の一度だけの補助スクリプト _upgrade_capacity.py、_finish_capacity.py、_finalize_capacity.py は履歴として保持し、再実行を無効にしています。診断や試験からは呼びません。"]
    return "\\n".join(lines)+"\\n"


def coil_sums''')
p.write_text(s)
