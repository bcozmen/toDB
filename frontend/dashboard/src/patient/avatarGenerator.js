/**
 * Avatar Generator using Dicebear Personas
 *
 * Rules:
 * - Age brackets:
 *   - Child: age < 18
 *   - Adult: 18 <= age < 60
 *   - Old: age >= 60
 *
 * - Clothes:
 *   - Child -> small
 *   - Female -> squared
 *   - Male -> rounded
 *
 * - Eyes:
 *   - Only glasses, happy, open
 *
 * - Facial hair:
 *   - Only for adult, old male (disabled for children and females)
 *
 * - Hair:
 *   - Male: bald, balding (only old male), buzzcut, curlyHighTop, fade, extraLong, mohawk, shortCombover
 *   - Female: bobBangs, bobCut, curly, curlyBun, curlyHighTop, extraLong, fade, long, pigtails (only child female), straightBun
 *
 * - Mouth:
 *   - smile, bigSmile
 *   - lips (only for adult/old woman)
 *   - pacifier (only for children)
 *
 * - Nose:
 *   - smallRound (for kids)
 *   - mediumRound (for adults)
 *   - wrinkles (for old)
 *
 * - Hair and facial hair color:
 *   - White (#ffffff) for old
 *
 * - Skin color by race & ethnicity
 */

const TODAY = 1790812800; // Simulated reference date (~2026)

export const DICEBEAR_PERSONAS_ENDPOINT = 'https://api.dicebear.com/10.x/personas/svg';

/**
 * Valid hair variants for Male
 */
export const MALE_HAIR_BASE = [
  'bald',
  'buzzcut',
  'curlyHighTop',
  'fade',
  'extraLong',
  'mohawk',
  'shortCombover',
];

export const MALE_HAIR_OLD = [
  'bald',
  'balding',
  'buzzcut',
  'curlyHighTop',
  'fade',
  'extraLong',
  'mohawk',
  'shortCombover',
];

/**
 * Valid hair variants for Female
 */
export const FEMALE_HAIR_BASE = [
  'bobBangs',
  'bobCut',
  'curly',
  'curlyBun',
  'curlyHighTop',
  'extraLong',
  'fade',
  'long',
  'straightBun',
];

export const FEMALE_HAIR_CHILD = [
  'bobBangs',
  'bobCut',
  'curly',
  'curlyBun',
  'curlyHighTop',
  'extraLong',
  'fade',
  'long',
  'pigtails',
  'straightBun',
];

/**
 * Allowed eyes variants
 */
export const ALLOWED_EYES = ['glasses', 'happy', 'open'];

/**
 * Allowed facial hair variants for adult/old male
 */
export const MALE_FACIAL_HAIR_VARIANTS = [
  'beardMustache',
  'goatee',
  'pyramid',
  'shadow',
  'soulPatch',
  'walrus',
];

/**
 * Get skin colors according to race & ethnicity
 */
export function getSkinColors(raceInput, ethnicityInput) {
  const race = String(raceInput || '').toLowerCase().trim();
  const ethnicity = String(ethnicityInput || '').toLowerCase().trim();

  const isHispanic = ethnicity.includes('hispanic') && !ethnicity.includes('non');

  if (race === 'black') {
    return ['623d36', '92594b', 'b16a5b'];
  }
  if (race === 'asian') {
    return ['eeb4a4', 'e7a391', 'e5a07e'];
  }
  if (race === 'hawaiian') {
    return ['d78774', 'b16a5b', 'e5a07e'];
  }
  if (race === 'native') {
    return ['b16a5b', 'd78774', '92594b'];
  }
  if (isHispanic) {
    return ['e5a07e', 'd78774', 'b16a5b'];
  }
  if (race === 'white') {
    return ['eeb4a4', 'e7a391'];
  }

  // 'other' or unspecified
  return ['eeb4a4', 'e7a391', 'e5a07e', 'd78774', 'b16a5b', '92594b', '623d36'];
}

/**
 * Get natural hair colors based on age, race, and ethnicity
 */
export function getHairColors(isOld, raceInput, ethnicityInput) {
  if (isOld) {
    return ['ffffff'];
  }

  const race = String(raceInput || '').toLowerCase().trim();
  const ethnicity = String(ethnicityInput || '').toLowerCase().trim();
  const isHispanic = ethnicity.includes('hispanic') && !ethnicity.includes('non');

  if (race === 'black' || race === 'asian' || race === 'native' || race === 'hawaiian') {
    return ['1a1a1a', '2c1b18', '362c47'];
  }
  if (isHispanic) {
    return ['1a1a1a', '2c1b18', '362c47', '4a3728', '6c4545'];
  }
  if (race === 'white') {
    return ['1a1a1a', '362c47', '4a3728', '6c4545', 'b55239', 'd6b370', 'f29c65'];
  }

  return ['1a1a1a', '362c47', '4a3728', '6c4545', 'd6b370'];
}

/**
 * Parse Unix seconds from date string or timestamp
 */
export function toUnixSeconds(value) {
  if (value === null || value === undefined || value === '') return null;
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? null : parsed / 1000;
}

/**
 * Calculate patient age in full years
 */
export function calculateAge(patient, refTimestamp = TODAY) {
  const birth = toUnixSeconds(patient?.patient_start_time ?? patient?.birthDate);
  const end = toUnixSeconds(patient?.patient_stop_time ?? patient?.deathDate);
  if (birth === null) return null;
  const refTime = end ?? refTimestamp;
  return Math.max(0, Math.floor((refTime - birth) / (365.2425 * 24 * 60 * 60)));
}

/**
 * Generate configuration options for Dicebear Personas
 */
export function getAvatarOptions({
  age = 35,
  race = 'other',
  ethnicity = 'nonhispanic',
  gender = 'M',
  seed = 'patient',
} = {}) {
  const numericAge = typeof age === 'number' && Number.isFinite(age) ? age : 35;
  const isChild = numericAge < 18;
  const isOld = numericAge >= 60;
  const isAdult = !isChild && !isOld;

  const g = String(gender || '').toUpperCase();
  const isFemale = g === 'F' || g === 'FEMALE';
  const isMale = !isFemale;

  // 1. Clothes
  // for child use small, for female squared, for male rounded
  const clothesVariant = isChild
    ? ['small']
    : isFemale
    ? ['squared']
    : ['rounded'];

  // 2. Eyes
  // only use glasses, happy, open
  const eyesVariant = ALLOWED_EYES;

  // 3. Facial hair
  // only for adult, old male
  const canHaveFacialHair = isMale && (isAdult || isOld);
  const facialHairProbability = canHaveFacialHair ? 40 : 0;
  const facialHairVariant = canHaveFacialHair ? MALE_FACIAL_HAIR_VARIANTS : [];

  // 4. Hair
  // Male -> Bald, Balding (only old male), Buzzcut, curlyHighTop, fade, extraLong, mohawk, shortCombover
  // Female -> bobBangs, bobCut, curly, curlyBun, curlyHighTop, extraLong, fade, long, pigtail (only child female), straightBun
  let hairVariant;
  if (isFemale) {
    hairVariant = isChild ? FEMALE_HAIR_CHILD : FEMALE_HAIR_BASE;
  } else {
    hairVariant = isOld ? MALE_HAIR_OLD : MALE_HAIR_BASE;
  }

  // 5. Mouth
  // can use smile, lips (only for woman), bigSmile, pacifier (only for childeren)
  const mouthVariant = ['smile', 'bigSmile'];
  if (isChild) {
    mouthVariant.push('pacifier');
  } else if (isFemale) {
    mouthVariant.push('lips');
  }

  // 6. Nose
  // smallRound (for kids), mediumRound (for adults), wrinkles (for old)
  let noseVariant;
  if (isChild) {
    noseVariant = ['smallRound'];
  } else if (isOld) {
    noseVariant = ['wrinkles'];
  } else {
    noseVariant = ['mediumRound'];
  }

  // 7. Colors
  const skinColor = getSkinColors(race, ethnicity);
  const hairColor = getHairColors(isOld, race, ethnicity);
  const facialHairColor = isOld ? ['ffffff'] : hairColor;

  return {
    seed: String(seed || 'patient'),
    clothesVariant,
    eyesVariant,
    facialHairProbability,
    ...(facialHairVariant.length > 0 ? { facialHairVariant } : {}),
    hairVariant,
    mouthVariant,
    noseVariant,
    skinColor,
    hairColor,
    facialHairColor,
  };
}

/**
 * Generate a Dicebear Personas SVG URL
 */
export function getAvatarUrl(params = {}) {
  const options = getAvatarOptions(params);
  const searchParams = new URLSearchParams();

  searchParams.append('seed', options.seed);

  const appendList = (key, list) => {
    if (!list) return;
    if (Array.isArray(list)) {
      for (const item of list) {
        if (item !== undefined && item !== null && item !== '') {
          searchParams.append(key, String(item));
        }
      }
    } else {
      searchParams.append(key, String(list));
    }
  };

  appendList('clothesVariant', options.clothesVariant);
  appendList('eyesVariant', options.eyesVariant);
  appendList('hairVariant', options.hairVariant);
  appendList('mouthVariant', options.mouthVariant);
  appendList('noseVariant', options.noseVariant);
  appendList('skinColor', options.skinColor);
  appendList('hairColor', options.hairColor);
  appendList('facialHairColor', options.facialHairColor);
  searchParams.append('facialHairProbability', String(options.facialHairProbability));

  if (options.facialHairVariant && options.facialHairVariant.length > 0) {
    appendList('facialHairVariant', options.facialHairVariant);
  }

  return `${DICEBEAR_PERSONAS_ENDPOINT}?${searchParams.toString()}`;
}

/**
 * Helper to build avatar URL directly from patient object
 */
export function getPatientAvatarUrl(patient, customSeed = null) {
  if (!patient) return null;
  const age = calculateAge(patient);
  const seed = customSeed || patient.patient_id || 'patient';

  return getAvatarUrl({
    age: age ?? 35,
    race: patient.race,
    ethnicity: patient.ethnicity,
    gender: patient.gender,
    seed,
  });
}
