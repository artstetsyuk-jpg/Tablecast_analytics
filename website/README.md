# TableCast website

The public landing page for TableCast: plain HTML, CSS and JavaScript, no build step,
no dependencies (only Google Fonts).

```
website/
  index.html          the page
  assets/css/style.css  look, layout, motion timing
  assets/js/main.js     skyline scene, parallax, scroll story, reveals, charts
  impressum.html        TEMPLATE, fill in before publishing
  privacy.html          TEMPLATE, fill in before publishing
```

## Preview on your computer

```bash
py -m http.server 8620 --directory website
```

Then open http://localhost:8620.

## Before publishing

1. **App link:** in `index.html`, set `window.TABLECAST_APP_URL` to your app's address.
   Every "Try the demo" / "Sign in" button uses it.
2. **Legal pages:** complete `impressum.html` and `privacy.html` (required in Austria/EU).
3. **Numbers:** the accuracy section quotes ~10% vs ~14% from a real 600-day bakery history.
   The chart beside it is marked as an illustration. Update them if your results change.

## Hosting (free)

Any static host works. With Netlify, Vercel or Cloudflare Pages, connect the GitHub repo and set the
publish directory to `website`. Then add your domain and register it in Google Search Console.

## How the cinematic effects work

- **Depth layers** in the hero (`[data-depth]`) move at different speeds with scroll and mouse,
  driven by the CSS variables `--p`, `--mx` and `--my` set in `main.js`.
- **The skylines** are drawn on `<canvas>` from a fixed random seed, so they look the same on every
  visit. Windows switch on during the intro and occasionally afterwards.
- **Film grain, letterbox bars and blurred foreground lights** give the film look.
- **Reveals** use `IntersectionObserver` plus CSS transitions with slow "camera" easing curves.
- **Reduced motion:** visitors who turn animations off in their system settings get a still,
  fully readable page.
