import { test, expect } from "@playwright/test";

const sizes = [
  { name: "desktop", width: 1440, height: 900 },
  { name: "tablet", width: 768, height: 1024 },
  { name: "mobile", width: 390, height: 844 },
  { name: "small-mobile", width: 320, height: 720 },
];

for (const size of sizes) {
  test(`TrustRAG responsive layout - ${size.name}`, async ({ page }) => {
    await page.setViewportSize({
      width: size.width,
      height: size.height,
    });

    await page.goto("http://127.0.0.1:5173");

    await expect(page.locator(".site-header")).toBeVisible();
    await expect(page.locator(".hero")).toBeVisible();
    await expect(page.locator(".library-layout")).toBeVisible();

    const dimensions = await page.evaluate(() => ({
      documentWidth: document.documentElement.scrollWidth,
      viewportWidth: window.innerWidth,
    }));

    expect(dimensions.documentWidth).toBeLessThanOrEqual(
      dimensions.viewportWidth
    );
  });
}
