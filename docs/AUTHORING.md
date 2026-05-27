# Authoring spells, rules, and class actions

This document specifies the YAML content the `wow-alert` app reads. It is
self-contained: an external assistant with no source-code access can use
this document to produce correct content files for a new dungeon or a new
class+spec.

---

## 1. What the app does (one paragraph)

The app watches WoW cast bars on screen. When it sees a cast that matches
an entry in your spell database, it consults a rule engine that decides
what to do — usually emit an audible callout. Rules can pick a
class-specific response (BoP, Cocoon, Kick, Cleanse, etc.) based on what
the player has off cooldown, what role the target plays, and whether a
target is visible at all. When no rule fires, the spell's default
`phrase` plays as a generic callout.

---

## 2. File layout

```
config/
  dungeons/
    _global.yaml                       # cross-dungeon spells + rules
    <dungeon_slug>.yaml                # one file per dungeon
  classes/
    <class>/<spec>.yaml                # one file per class+spec
```

`<dungeon_slug>` is the dungeon's display name lowercased with non-alnum
characters collapsed to underscores — `"Mists of Tirna Scithe"` →
`mists_of_tirna_scithe.yaml`. The display name lives inside the file's
`dungeon:` header so the slug is purely a filesystem convention.

`<class>` and `<spec>` use lowercase + underscore. Allowed classes:

```
death_knight, demon_hunter, druid, evoker, hunter, mage, monk,
paladin, priest, rogue, shaman, warlock, warrior
```

Allowed specs per class:

```
death_knight  : blood, frost, unholy
demon_hunter  : havoc, vengeance
druid         : balance, feral, guardian, restoration
evoker        : devastation, preservation, augmentation
hunter        : beast_mastery, marksmanship, survival
mage          : arcane, fire, frost
monk          : brewmaster, mistweaver, windwalker
paladin       : holy, protection, retribution
priest        : discipline, holy, shadow
rogue         : assassination, outlaw, subtlety
shaman        : elemental, enhancement, restoration
warlock       : affliction, demonology, destruction
warrior       : arms, fury, protection
```

---

## 3. The class action library

### Purpose

The class library describes the player's abilities along three abstract
dimensions so rules can request *category* (e.g. "any defensive") instead
of a specific spell name. This lets one rule work for every class — what
"single-target defensive" means is BoP for paladin, Cocoon for monk,
Ironbark for druid.

### File: `config/classes/<class>/<spec>.yaml`

```yaml
class: paladin                     # must equal the parent directory name
spec: holy                         # must equal the filename without .yaml

actions:
  - id: blessing_of_protection     # stable id; referenced by rules
    label: "BOP"                   # short TTS-friendly callout name
    category: defensive            # controlled vocab; see below
    scope: single_target           # controlled vocab; see below
    tags: [aggro_dropping]         # free-form list; rules filter via has_tag/lacks_tag
    spell_id: 1022                 # canonical WoW spell ID; see the icon DB section
```

### Field reference

**`id`** — Unique within the file. Lowercase + underscore. Referenced by
`{action.id}` in rule templates.

**`label`** — The word(s) spoken aloud when this action fires. Should be
brief (1–3 syllables). Examples: `"BOP"`, `"Sac"`, `"Cocoon"`,
`"Lay on Hands"`, `"Rebuke"`. The TTS clip is prerendered at calibration
time.

**`category`** — One of:

| Category | Meaning | Examples |
|---|---|---|
| `defensive` | Reduces incoming damage on someone | BoP, Sac, Cocoon, Ironbark, Aura Mastery |
| `heal` | Restores health | Lay on Hands, Revival, Tranquility |
| `dispel` | Removes harmful effects | Cleanse, Purge, Remove Curse, Mass Dispel |
| `interrupt` | Interrupts an enemy cast | Rebuke, Counterspell, Kick, Mind Freeze |
| `cc` | Crowd-controls an enemy | Hammer of Justice, Hex, Polymorph |
| `stop` | Stops an effect via knockback / displacement / silence | Shockwave, Typhoon, Silence |

**`scope`** — Who the action affects:

| Scope | Meaning |
|---|---|
| `self` | Only the player |
| `single_target` | One party member (the caller picks who) |
| `party_wide` | Everyone in the party |
| `raid_wide` | Everyone in the raid |

**`tags`** — Free-form list of strings. Rules filter on tags via
`has_tag` and `lacks_tag` predicates. Common tags:

| Tag | Meaning |
|---|---|
| `aggro_dropping` | Drops the target's threat — don't suggest on a tank |
| `emergency_only` | Reserve for crises (Lay on Hands, Defile) |
| `mana_intensive` | High mana cost; deprioritize when mana-stressed |
| `positions_target` | Forces target to a position (Death Grip, Body and Soul) |
| `magic`, `poison`, `disease`, `curse` | Dispel subtypes (Cleanse has `[magic, poison, disease]`; mage Remove Curse has `[curse]`) |
| `requires_los` | Needs line of sight |
| `melee_range` | Caster must be in melee range |

Invent new tags freely — the engine treats them as opaque strings.

**`spell_id`** — The canonical WoW spell ID for this ability. Acts as
the join key to the icon database and the cooldown availability dict.
You can look spell IDs up on Wowhead — the URL `https://www.wowhead.com/spell=1022`
is for Blessing of Protection, so `spell_id: 1022`.

### The cooldown manager

The app tracks which of the player's abilities are currently usable.
This is core to authoring: a rule never has to ask "is this spell off
cooldown?" — the engine answers that automatically and skips actions
that aren't available.

How it works end-to-end:

1. **Icon database** lives at `config/icons/<spell_id>.png`. One PNG
   per spell that any class library references. Populate it by running
   `python -m wow_alert.tools.fetch_icons`, which scrapes Wowhead for
   every spell_id declared in `config/classes/*/*.yaml` and writes the
   icon. Idempotent — re-run after adding new actions to pick up only
   the new icons.
2. **Calibration** uses a vision LLM to locate the bounding box of the
   cooldown manager region and the individual icon bboxes inside it.
   It does NOT try to name the icons.
3. **Icon matcher** (`wow_alert/icon_matcher.py`) then template-matches
   each calibrated icon bbox against every PNG in `config/icons/` and
   assigns a `spell_id` to the icon (or None when no reference scores
   above the confidence threshold). This step happens locally on CPU
   right after calibration — no extra LLM cost.
4. **Class+spec auto-detection.** The app counts which icon spell_ids
   appear in each `config/classes/<class>/<spec>.yaml` file and picks
   the spec with the most matches. Replaces the previous LLM "guess
   the class" pass, which was unreliable.
5. **Cooldown watcher** runs at ~2 FPS. For each calibrated icon bbox
   whose `spell_id` is set, it samples the pixels and writes
   `cooldowns[spell_id] = True/False`. Icons that didn't match anything
   (spell_id is None) are skipped — there's no key to write under.
6. **Rule engine** iterates the class library in file order whenever a
   priority carries a class-action filter (`category` / `scope` /
   `has_tag` / `lacks_tag`). For each candidate action it checks
   `cooldowns[action.spell_id]`:
   - False → available, action binds.
   - True → on cooldown, skip this action, continue iterating.
   - **Missing → treat as on cooldown (fail-closed).** Untracked
     actions never bind. The rule walker continues to the next
     priority; if every priority fails, the engine falls through to
     the spell's default `phrase`.
7. **Startup warning.** If a loaded class-library action has a
   `spell_id` that doesn't appear in the calibrated icon set, the app
   logs a WARNING at calibration time. The action is untracked, so
   rules that would bind it will fall through to the spell's default
   phrase instead. Fix by either adding the ability to your in-game
   cooldown manager and recalibrating, or removing the action from
   the class library.

The practical consequence for authors: **a rule that asks for "any
single-target defensive" will naturally cycle through BoP → Sac →
whatever else is in the library, picking whichever is up.** You don't
need to enumerate "if BoP on cooldown then Sac" — just list both
defensives in the library (in preferred order) and write one rule that
asks for `category: defensive, scope: single_target`.

The `spell_id` field on each class action is the canonical join key.
The icon matcher resolves on-screen icons to spell IDs locally (no LLM
involved in this step), so as long as `config/icons/<spell_id>.png`
exists and the in-game ability is on your cooldown bar, the join works.

### Authoring workflow

1. Add or edit an action in `config/classes/<class>/<spec>.yaml`. Set
   `spell_id:` to the WoW spell ID (look it up on Wowhead — the URL is
   `https://www.wowhead.com/spell=<id>`).
2. Run `python -m wow_alert.tools.fetch_icons`. Idempotent — fetches
   only the icons missing from `config/icons/`.
3. Recalibrate in-game so the matcher runs against your bar.

### Complete example: Holy Paladin

```yaml
class: paladin
spec: holy

actions:
  - id: blessing_of_protection
    label: "BOP"
    category: defensive
    scope: single_target
    tags: [aggro_dropping]
    spell_id: 1022

  - id: blessing_of_sacrifice
    label: "Sac"
    category: defensive
    scope: single_target
    spell_id: 6940

  - id: lay_on_hands
    label: "Lay on Hands"
    category: heal
    scope: single_target
    tags: [emergency_only]
    spell_id: 633

  - id: devotion_aura
    label: "Devotion Aura"
    category: defensive
    scope: party_wide
    spell_id: 465

  - id: aura_mastery
    label: "Aura Mastery"
    category: defensive
    scope: party_wide
    spell_id: 31821

  - id: cleanse
    label: "Cleanse"
    category: dispel
    scope: single_target
    tags: [magic]
    spell_id: 4987

  - id: hammer_of_justice
    label: "Hammer of Justice"
    category: cc
    scope: single_target
    spell_id: 853
```

### Same shape for a different class — Mistweaver Monk

```yaml
class: monk
spec: mistweaver

actions:
  - id: life_cocoon
    label: "Cocoon"
    category: defensive
    scope: single_target
    spell_id: 116849

  - id: revival
    label: "Revival"
    category: heal
    scope: party_wide
    tags: [emergency_only]
    spell_id: 115310

  - id: detox
    label: "Detox"
    category: dispel
    scope: single_target
    tags: [magic]
    spell_id: 218164
```

---

## 4. The dungeon file: spells

### Purpose

Each spell entry tells the app "this cast bar text exists, react like
this". Fuzzy matching handles OCR jitter — you don't have to enumerate
every misread.

### File: `config/dungeons/<dungeon_slug>.yaml`

```yaml
dungeon: "Windrunner Spire"       # display name; omit only in _global.yaml

spells:
  - id: windrunner_spire_spirit_bolt
    name: "Spirit Bolt"
    aliases: ["Spirit Boit", "Spirlt Bolt"]
    severity: danger
    phrase: "KICK SPIRIT BOLT"
    duration: 2.5
    cast_by: mob
    school: shadow
    interruptible: true
    notes: "Restless Steward filler cast."

rules: []
```

### Field reference

**`id`** — Unique stable identifier across all dungeon files. Convention:
`<dungeon_slug>_<spell_slug>`. Used by rules' `on_cast.spell_id`.

**`name`** — The canonical spell name as displayed on the cast bar.
Primary fuzzy-match target.

**`aliases`** — Known OCR misreads of `name`. Each alias also matches
the spell at lookup time. Common forms to anticipate: character-level
substitutions (`Boit` ↔ `Bolt`, `lt` ↔ `It`), merged spaces
(`SpiritBolt`), or repeated letters.

**`severity`** — One of:

| Severity | Behavior |
|---|---|
| `danger` | Significant threat; alert plays |
| `info` | Informational; alert plays |
| `ignore` | Recognized but suppressed; no alert |

Rules can fire for any severity. `ignore` only changes the no-rule
fallback (which becomes silent).

**`phrase`** — Default TTS phrase. Played when no rule matches the spell,
or when a matching priority has no `do:` (it's an Alert, not a
Recommendation). Keep it short, action-oriented, often in caps for
emphasis: `"KICK NOW"`, `"DEFENSIVE ARCANE"`, `"DODGE BREATH"`,
`"BIG AOE"`, `"TANK BUSTER"`. The phrase is prerendered as a single
WAV clip.

**`duration`** — Cast time in seconds. Optional. When set, used as the
dedupe TTL so the same in-flight cast doesn't re-alert. Recommended
whenever you know the value; OCR durations are unreliable.

**`cast_by`**, **`school`**, **`interruptible`**, **`notes`** —
Descriptive metadata for human readers. The engine does not consume
these. Useful for keeping authors honest about which spells are
actually interruptible, which schools to dispel, etc.

---

## 5. The dungeon file: rules

### Purpose

A rule says "when *this* spell is cast, and *these conditions* hold, do
*this*." Without a rule, a spell just emits its default `phrase` as an
Alert. With a rule, you can pick a class-specific action, branch on
target role, fall through to alternatives, or override the spoken phrase.

### Schema

```yaml
rules:
  - on_cast:
      spell_id: windrunner_spire_arcane_salvo   # required
      target_role: tank                         # optional filter

    priorities:
      # First priority whose conditions all pass wins.
      - category: defensive          # class-action filter
        scope: single_target         # class-action filter
        lacks_tag: aggro_dropping    # class-action filter
        target_role: tank            # cast filter
        target_present: true         # cast filter
        say: "{action.label} {target}"
        do: "{action.id}"
        phrase: "Break Shield"
```

### `on_cast` block

**`spell_id`** (required) — The id of the spell that triggers this rule.
Must match a `spells.id` somewhere in your loaded config.

**`target_role`** (optional) — Restricts the rule to casts whose target
plays this role. One of `tank` / `healer` / `dps`. Roles are assigned
during calibration and stored per roster member.

A rule's *specificity* is the count of `on_cast` filters it sets (today
just `target_role` counts). More specific rules sort higher when two
rules match the same cast — so a tank-specific rule wins over a generic
one whenever the target is the tank, and the generic rule fires for
non-tank targets.

### `priorities` list

Evaluated top to bottom. The first priority whose conditions all pass
fires. Subsequent priorities in the same rule are ignored.

A priority is a flat object: zero or more **condition** fields plus the
**output** fields. All set conditions must pass (AND). A priority with
no condition fields is the catch-all.

| Field | Kind | Purpose |
|---|---|---|
| `category` | condition (class-action) | Required action category — `defensive`/`heal`/`dispel`/`interrupt`/`cc`/`stop` |
| `scope` | condition (class-action) | Required action scope — `self`/`single_target`/`party_wide`/`raid_wide` |
| `has_tag` | condition (class-action) | Action must carry this tag |
| `lacks_tag` | condition (class-action) | Action must NOT carry this tag |
| `target_role` | condition (cast) | Cast's target plays this role |
| `target_present` | condition (cast) | `true` = require a target, `false` = require no target |
| `say` | output | Templated message string (always required) |
| `do` | output | Templated action id; when set, output is a Recommendation |
| `phrase` | output | Templated audio override; only meaningful with `do` |

### Conditions

Two kinds of conditions, evaluated AND.

#### Class-action filters

Setting any of `category` / `scope` / `has_tag` / `lacks_tag` tells the
engine to **bind a class action** for this priority. It walks the
player's class library in file order and picks the first action that:

1. Has the requested `category` (if set)
2. Has the requested `scope` (if set)
3. Has the `has_tag` tag (if set)
4. Does NOT have the `lacks_tag` tag (if set)
5. **Is not currently on cooldown.** Looked up via the cooldown manager
   (see §3 — "The cooldown manager"). An action is on cooldown when
   `cooldowns[action.spell_id]` is True OR when the spell_id is missing
   from the dict (fail-closed — untracked actions don't bind). **Skips
   are silent** — iteration continues to the next library entry. The
   startup warning lists untracked actions so the gap is visible
   without you having to notice missing recommendations.

If no action passes all of these, the priority fails and the next
priority is tried. When an action binds it becomes available to
templates as `{action.label}` and `{action.id}`.

If multiple actions satisfy the filters, the **first one in file order**
wins. Put higher-priority actions earlier in the class library file —
the cooldown filter will automatically fall through to the next entry
when the preferred one is unavailable.

A priority with no class-action filter does no binding; `{action.*}`
tokens in its templates would render as literal text, so omit them.

#### `target_role`

```yaml
target_role: tank   # or healer, dps
```

Matches when the cast's canonical target has the specified role. Fails
if the target's role is unknown.

#### `target_present`

```yaml
target_present: true    # or false
```

Matches when the cast does (true) or does not (false) have a target.
Use `true` to suppress priorities whose templates only make sense with
a named target (e.g., `"{action.label} {target}"` would render as just
the bare action label without one).

### Templating

These tokens are replaced in `say`, `do`, and `phrase`. Whitespace is
collapsed and trimmed after substitution.

| Token | Replacement |
|---|---|
| `{target}` | Canonical roster name; or raw OCR target; or empty |
| `{spell}` | The spell's canonical `name` |
| `{duration}` | `"3.0s"`; or empty when no duration |
| `{action.label}` | The bound action's `label` (set only when the priority has a class-action filter) |
| `{action.id}` | The bound action's `id` (set only when the priority has a class-action filter) |

### Output behavior

| `do:` set? | Output | Audio | Log message |
|---|---|---|---|
| No  | `Alert` | spell's `phrase` (from spells entry) | rendered `say` |
| Yes | `Recommendation` | rendered `phrase` if set, else action.label; stitched with target name | rendered `say` |

When the engine emits a `Recommendation` with a target, audio plays as a
stitched clip: `<phrase>.wav` + `<target_name>.wav` concatenated. Both
clips must exist in the prerender cache (they're generated at
calibration time from the class library labels, rule literal phrases,
and roster names).

### Fallback

If no rule matches the spell, or no priority within a matched rule
fires, the engine emits an Alert with the spell's default `phrase`. So
authoring rules is purely *additive* — without rules, every matched
spell still plays its default callout.

---

## 6. How decisions are made (worked example)

A cast comes in: `Spell="Arcane Salvo", Target="Captain Garrick"`.
Calibration has assigned `roles["Captain Garrick"] = "tank"`.

Loaded rules (abbreviated):

```yaml
# Rule A — specific to tank targets
- on_cast: { spell_id: arcane_salvo, target_role: tank }
  priorities:
    - category: defensive
      scope: single_target
      lacks_tag: aggro_dropping
      say: "{action.label} {target}"
      do: "{action.id}"
    - category: defensive
      scope: party_wide
      say: "{action.label}"
      do: "{action.id}"
    - say: "Tank Buster on {target}"

# Rule B — generic
- on_cast: { spell_id: arcane_salvo }
  priorities:
    - category: defensive
      scope: single_target
      target_present: true
      say: "{action.label} {target}"
      do: "{action.id}"
    - category: defensive
      scope: party_wide
      target_present: false
      say: "{action.label}"
      do: "{action.id}"
```

Loaded class library (paladin/holy, abbreviated):

```yaml
actions:
  - id: bop                       # tagged aggro_dropping
    label: BOP
    category: defensive
    scope: single_target
    tags: [aggro_dropping]
  - id: blessing_of_sacrifice     # untagged
    label: Sac
    category: defensive
    scope: single_target
  - id: devotion_aura
    label: Devotion Aura
    category: defensive
    scope: party_wide
```

Engine flow:

1. **Rule selection**: both rules match `spell_id`. Rule A has
   `target_role: tank` set → specificity 1. Rule B has no filters →
   specificity 0. Rule A sorts first.
2. **Rule A priority 1** — class-action filter `{ defensive,
   single_target, lacks_tag: aggro_dropping }`:
   - Walk class actions. BoP: matches category+scope, has
     `aggro_dropping` → `lacks_tag` filter excludes it. Sac: matches
     category+scope, has no excluded tag, cooldown OK → **bind action
     = blessing_of_sacrifice**.
3. **Build output**: `do` is set → `Recommendation`.
   - `say` renders `"Sac Captain Garrick"`.
   - `phrase` defaults to `action.label = "Sac"`.
   - `target = "Captain Garrick"`.
4. **Audio**: `Sac.wav` + `Captain_Garrick.wav` stitched and played.

If Sac were on cooldown, Rule A priority 1 would fail (no untagged
defensive available), and priority 2 (`party_wide` defensive) would
match Devotion Aura. If both were down, priority 3 (catch-all `say`)
would fire as an Alert reading `"Tank Buster on Captain Garrick"` with
the spell's default audio phrase.

If the target were Meredy (dps) instead, Rule A's `target_role: tank`
filter would fail, Rule B would fire instead, BoP would match (no tag
exclusion in Rule B), and the result would be `"BOP Meredy"`.

If no target were detected, Rule A's `target_role: tank` filter would
fail (role lookup on null target). Rule B's first priority requires
`target_present: true` and is skipped. Rule B's second priority asks
for a party-wide defensive with `target_present: false` — Devotion Aura
matches and a Recommendation fires speaking `"Devotion Aura"`. If
party-wide defensives are also on cooldown, the engine falls through to
the spell-default Alert (`"DEFENSIVE ARCANE"`).

---

## 7. Common patterns (cookbook)

### Interrupt a cast

```yaml
- on_cast:
    spell_id: <spell>
  priorities:
    - category: interrupt
      say: "{action.label} {target}"
      do: "{action.id}"
# Falls through to spell's `phrase` when no interrupt available.
```

### Magic dispel

```yaml
- on_cast:
    spell_id: <spell>
  priorities:
    - category: dispel
      has_tag: magic
      say: "{action.label} {target}"
      do: "{action.id}"
```

### Tank buster — external defensive (avoid aggro drop on tank)

```yaml
- on_cast:
    spell_id: <spell>
    target_role: tank
  priorities:
    - category: defensive
      scope: single_target
      lacks_tag: aggro_dropping
      say: "{action.label} {target}"
      do: "{action.id}"
    - category: defensive
      scope: party_wide
      say: "{action.label}"
      do: "{action.id}"
    - say: "Tank Buster on {target}"
```

### Generic targeted damage — any defensive on the target

```yaml
- on_cast:
    spell_id: <spell>
  priorities:
    - category: defensive
      scope: single_target
      target_present: true
      say: "{action.label} {target}"
      do: "{action.id}"
    # OCR sometimes loses the target name mid-cast. Fall back to a
    # party-wide defensive so the player still gets a useful callout
    # ("Devotion Aura" / "Aura Mastery" / equivalent for their class).
    - category: defensive
      scope: party_wide
      target_present: false
      say: "{action.label}"
      do: "{action.id}"
# If both branches fail, the spell's default `phrase` fires.
```

### Custom phrase that doesn't match any single action

E.g., the action is `rebuke` (an interrupt) but the *meaningful* callout
is "Break Shield" because the cast puts up an absorb shield:

```yaml
- on_cast:
    spell_id: <spell>
  priorities:
    - category: interrupt
      say: "Break Shield"
      do: "{action.id}"
      phrase: "Break Shield"        # speaks "Break Shield" instead of "Rebuke"
    - category: defensive
      scope: party_wide
      say: "{action.label}"
      do: "{action.id}"
```

Literal `phrase` strings get auto-prerendered. Templated phrases
(containing `{...}`) compose at playback from already-cached clips.

### Hard-to-react spell — just warn

When the player can't directly help, omit `do:` everywhere and lean on
the spell's `phrase`. No rule needed at all if the default `phrase` is
enough.

### Severity ignore — recognize but stay silent

For frequent cast bars you don't want to hear about (e.g., trash filler
the tank handles), set `severity: ignore` on the spell. The engine logs
the registration but emits no Alert. Rules still apply if authored.

---

## 8. Pitfalls

**`phrase` strings are spoken aloud verbatim.** Keep them short and
pronounceable. `"KICK NOW"` works; `"You should kick this spell quickly
before it lands"` does not.

**`action.label` strings are spoken aloud verbatim.** Use short forms:
`"Sac"` instead of `"Blessing of Sacrifice"`, `"AM"` if `"Aura Mastery"`
trips up TTS, etc.

**`spell_id` joins through the local icon DB.** The cooldown watcher
writes `cooldowns[spell_id]`. The matcher only assigns a spell_id to a
calibrated icon if it found a matching PNG in `config/icons/`. If the
PNG is missing, the icon stays unidentified and the action looks
always-available to the rule engine. Symptom is a recommendation that
keeps firing. Fix: run `python -m wow_alert.tools.fetch_icons`. The
calibration apply step also logs a per-action WARNING for any
unmatched spell_id at the time the calibration is loaded.

**`target_role` only matches confidently-assigned roles.** If
calibration didn't assign a role for a roster member (the LLM was
uncertain and the user didn't override), `target_role` conditions fail.
Roles are: `tank`, `healer`, `dps`.

**Class-action binding returns the first matching action in file order.**
If you want one defensive to be preferred over another in the same
category, put the preferred one earlier in the class library file —
not the other way around in the rule.

**Rule specificity considers only `on_cast` filters.** A rule with
`target_role: tank` beats a rule without one, but two rules at the same
specificity fire in file order (first one in the YAML wins). Order your
generic rules after your specific rules.

**Aliases are not regex.** Each alias is matched literally (with the
same fuzzy threshold as the canonical name). If OCR consistently
mangles a name in a specific way, add that specific mangled form as an
alias.

**Don't rely on `cast_by` / `school` / `interruptible` / `notes`.**
These are descriptive only — the engine doesn't read them.

---

## 9. Complete worked example: Spellguard Magus / Arcane Salvo

Goal: when Spellguard Magus casts Arcane Salvo (a targeted-damage spell
on a party member), play a class-specific defensive on the target. Avoid
BoP on a tank because it drops aggro. Fall back to party-wide defensive,
then to the spell's generic warning.

### Spell entry (in `config/dungeons/windrunner_spire.yaml`)

```yaml
- id: windrunner_spire_arcane_salvo
  name: "Arcane Salvo"
  aliases: ["Arcane Saivo", "Arcane Salvoe"]
  severity: danger
  phrase: "DEFENSIVE ARCANE"
  duration: 5.0
  cast_by: mob
  school: arcane
  interruptible: false
  notes: "Spellguard Magus targeted damage. Use defensive if targeted."
```

### Rules (in the same file)

```yaml
- on_cast:
    spell_id: windrunner_spire_arcane_salvo
    target_role: tank
  priorities:
    - category: defensive
      scope: single_target
      lacks_tag: aggro_dropping
      say: "{action.label} {target}"
      do: "{action.id}"
    - category: defensive
      scope: party_wide
      say: "{action.label}"
      do: "{action.id}"
    - say: "Tank Buster on {target}"

- on_cast:
    spell_id: windrunner_spire_arcane_salvo
  priorities:
    - category: defensive
      scope: single_target
      target_present: true
      say: "{action.label} {target}"
      do: "{action.id}"
    - category: defensive
      scope: party_wide
      target_present: false
      say: "{action.label}"
      do: "{action.id}"
```

### Class library snippet (relevant Holy Paladin entries)

```yaml
- id: blessing_of_protection
  label: "BOP"
  category: defensive
  scope: single_target
  tags: [aggro_dropping]
  spell_id: 1022
- id: blessing_of_sacrifice
  label: "Sac"
  category: defensive
  scope: single_target
  spell_id: 6940
- id: devotion_aura
  label: "Devotion Aura"
  category: defensive
  scope: party_wide
  spell_id: 465
```

### Resulting behavior

| Cast | Outcome |
|---|---|
| Arcane Salvo on tank, Sac up | Recommendation: speaks `"Sac <tank>"` |
| Arcane Salvo on tank, Sac on CD, Devo Aura up | Recommendation: speaks `"Devotion Aura"` |
| Arcane Salvo on tank, all defensives on CD | Alert: speaks `"DEFENSIVE ARCANE"`, log says `"Tank Buster on <tank>"` |
| Arcane Salvo on DPS, BoP up | Recommendation: speaks `"BOP <dps>"` |
| Arcane Salvo with no target, Devo Aura up | Recommendation: speaks `"Devotion Aura"` |
| Arcane Salvo with no target, Devo on CD, Aura Mastery up | Recommendation: speaks `"Aura Mastery"` |
| Arcane Salvo with no target, all party-wide defensives on CD | Alert: speaks `"DEFENSIVE ARCANE"` |

---

## 10. Authoring checklist

When adding a new dungeon or class+spec:

- [ ] Choose the dungeon's display name; compute its slug (lowercase, underscores)
- [ ] Create `config/dungeons/<slug>.yaml` with the `dungeon:` header
- [ ] List every dangerous / interruptible / dispellable cast as a spell entry
- [ ] Author 2–3 aliases per spell to cover common OCR misreads
- [ ] Set `severity` (`danger` / `info` / `ignore`) and a short `phrase`
- [ ] Author rules where you want class-specific behavior
- [ ] For each class+spec the dungeon will be played on, ensure
      `config/classes/<class>/<spec>.yaml` exists with the relevant
      actions (interrupt, dispels, defensives) and a `spell_id:` on each
- [ ] Run `python -m wow_alert.tools.fetch_icons` to populate
      `config/icons/<spell_id>.png` for every new spell_id
- [ ] Tag actions appropriately (`aggro_dropping`, `magic`, etc.) so
      rule predicates can filter them
- [ ] For tank busters, write a tank-specific rule with
      `target_role: tank` and `lacks_tag: aggro_dropping`
- [ ] For targeted damage on anyone, write a generic rule with
      `target_present: true` so it falls through cleanly when OCR
      misses the target
- [ ] For mechanics where a *specific phrase* is more useful than the
      action's name (e.g., "Break Shield"), use `phrase:` override
