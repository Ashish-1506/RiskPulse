"""Automated 1920x1080 HD screenshot capture script for RiskPulse demonstration.

Connects to running Streamlit (http://localhost:8501) and FastAPI (http://127.0.0.1:8000)
instances and saves presentation screenshots into docs/results/.
"""

import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options

OUTPUT_DIR = Path("docs/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def init_driver() -> webdriver.Edge:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Edge(options=options)
    driver.set_window_size(1920, 1080)
    return driver


def wait_for_streamlit(driver, wait_seconds: float = 3.0):
    time.sleep(wait_seconds)


def capture_all():
    driver = init_driver()
    try:
        # 1. API Docs
        print("Capturing Screenshot 3: API Docs...")
        driver.get("http://127.0.0.1:8000/docs")
        time.sleep(2.0)
        driver.save_screenshot(str(OUTPUT_DIR / "screenshot_3_api_docs.png"))
        print("  -> Saved docs/results/screenshot_3_api_docs.png")

        # 2. Live Signals Page (Default overview)
        print("Capturing Screenshot 1: Live Risk Signals...")
        driver.get("http://localhost:8501/Live_Risk_Signals")
        wait_for_streamlit(driver, 4.0)
        driver.save_screenshot(str(OUTPUT_DIR / "screenshot_1_live_signals.png"))
        print("  -> Saved docs/results/screenshot_1_live_signals.png")

        # 3. Live Signals Page with Headline Injection
        print("Capturing Screenshot 2: Headline Injection...")
        try:
            # Click expander if closed
            expanders = driver.find_elements(By.CSS_SELECTOR, "[data-testid='stExpander'] summary")
            if expanders:
                expanders[0].click()
                time.sleep(1.0)

            # Click Scenario 1 (Geopolitical Shock)
            buttons = driver.find_elements(By.TAG_NAME, "button")
            for b in buttons:
                if "Geopolitical Shock" in b.text:
                    b.click()
                    time.sleep(1.0)
                    break

            # Click Run Analysis
            buttons = driver.find_elements(By.TAG_NAME, "button")
            for b in buttons:
                if "Run Analysis" in b.text:
                    b.click()
                    time.sleep(2.5)
                    break
        except Exception as e:
            print(f"  Note during injection interaction: {e}")

        driver.save_screenshot(str(OUTPUT_DIR / "screenshot_2_headline_injection.png"))
        print("  -> Saved docs/results/screenshot_2_headline_injection.png")

        # 4. Strategic Stress Test - Shocked Portfolio (After)
        print("Capturing Screenshot 5: Wholesale Stress Test (Shocked)...")
        driver.get("http://localhost:8501/Strategic_Stress_Test")
        wait_for_streamlit(driver, 4.0)
        driver.save_screenshot(str(OUTPUT_DIR / "screenshot_5_stress_test_after.png"))
        print("  -> Saved docs/results/screenshot_5_stress_test_after.png")

        # 5. Strategic Stress Test - Unshocked (Before)
        print("Capturing Screenshot 4: Wholesale Stress Test (Unshocked)...")
        try:
            # Adjust slider or mode to unshocked
            radios = driver.find_elements(By.CSS_SELECTOR, "[data-testid='stRadio'] label")
            for r in radios:
                if "Manual scenario" in r.text:
                    r.click()
                    time.sleep(1.5)
                    break
        except Exception as e:
            print(f"  Note during unshocked interaction: {e}")
        driver.save_screenshot(str(OUTPUT_DIR / "screenshot_4_stress_test_before.png"))
        print("  -> Saved docs/results/screenshot_4_stress_test_before.png")

        # 6. Tactical Index Rebalancer - Weights trajectory
        print("Capturing Screenshot 6: Tactical Weights Allocation Over Time...")
        driver.get("http://localhost:8501/Tactical_Index_Rebalancer")
        wait_for_streamlit(driver, 4.0)
        driver.save_screenshot(str(OUTPUT_DIR / "screenshot_6_rebalancer_weights.png"))
        print("  -> Saved docs/results/screenshot_6_rebalancer_weights.png")

        # 7. Tactical Index Rebalancer - Backtest curve (scrolled down)
        print("Capturing Screenshot 7: Backtest Equity Curve...")
        try:
            driver.execute_script("window.scrollTo(0, 1000);")
            time.sleep(1.0)
        except Exception:
            pass
        driver.save_screenshot(str(OUTPUT_DIR / "screenshot_7_backtest_curve.png"))
        print("  -> Saved docs/results/screenshot_7_backtest_curve.png")

        print("\nAll 7 high-resolution screenshots captured successfully!")

    finally:
        driver.quit()


if __name__ == "__main__":
    capture_all()
