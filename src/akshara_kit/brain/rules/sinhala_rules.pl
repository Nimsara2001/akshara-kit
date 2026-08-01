% Sinhala morphological rule base for micro-chunk boundary detection.
% Realises interim report Section 6.5.1.
%
% Source of the linguistics
% -------------------------
% Every fact below is taken from:
%   වියරණ විවරණ (අක්ෂර, සන්ධි හා වාක්‍ය රීති), හෙ. ව. බිහේෂ් ඉන්දික සම්පත්,
%   කැලණිය විශ්වවිද්‍යාලය, 2013 — §4 උක්ත ආඛ්‍යාත සම්බන්ධතා, pp. 89-112.
% Page references appear against each group so a linguist can check the rule
% against the source rather than against the code.
%
% The organising fact (p.89)
% --------------------------
% Sinhala is SOV: කර්තෘ - කර්මය - ක්‍රියාව. The *final* verb of a sentence is its
% ආඛ්‍යාතය (predicate) — "වාක්‍යයේ අවසාන ක්‍රියාව ආඛ්‍යාතය නමින් හැඳින් වේ".
% So a finite verb ending IS a sentence boundary. This matters because Sinhala
% prose uses terminal punctuation sparingly; keying only on '.' as the prototype
% did misses most real sentence ends.
%
% Four tiers, deliberately separate
% ---------------------------------
%   1. sentence_terminator/1  - ends a sentence   (finite ආඛ්‍යාත forms)
%   2. clause_boundary/1      - ends a clause, sentence continues (non-finite)
%   3. discourse_connective/1 - starts a new thought
%   4. never_split/1          - binds within one clause; splitting breaks meaning
%
% Conflating 1 and 2 is the prototype's main flaw: ලා, මින්, ගොස් are conjunctive
% participles (පූර්වක්‍රියා). They close a clause while the sentence runs on to
% its finite verb. Treating them as sentence ends yields fragments with no
% predicate.

% Every atom below is Sinhala, so the source encoding must be declared. Without
% this the file still loads under a consult from the Machine Query Interface,
% but the atoms are read in the platform encoding and no Sinhala rule ever
% matches — the rule base silently degrades to punctuation-only splitting.
:- encoding(utf8).

:- module(sinhala_rules, [
       check_split/3,
       check_split/2,
       is_sentence_end/1,
       is_clause_end/1,
       is_never_split/1
   ]).

% ===================================================================
% TIER 1 - Sentence terminators: the finite predicate (ආඛ්‍යාතය)
% ===================================================================

% ශුද්ධ ආඛ්‍යාතය - the verb used bare. Person/number endings, p.94-96.
% A pure predicate inflects for tense, person, voice and number but NOT gender.
%
%   1sg  මම කරමි / කළෙමි / කරන්නෙමි          -මි
%   1pl  අපි යමු / ගියෙමු / යන්නෙමු           -මු
%   2sg  තෝ යහි / ගියෙහි / යන්නෙහි           -හි
%   2pl  තොපි යහු / ගියෙහු / යන්නෙහු          -හු
%   3sg  ඔහු කරයි / යයි / වැටෙයි              -යි
%   3pl  ඔවුහු යති / කරති / පවසති              -ති
%   3pl  past කළහ / වූහ / වැඩියහ / ලදහ         -හ
finite_ending('මි').
finite_ending('මු').
finite_ending('හි').
finite_ending('හු').
finite_ending('යි').
finite_ending('ති').
finite_ending('හ').

% KNOWN AMBIGUITY - හි and හු over-fire on nouns.
%
% -හි is both the 2sg verb ending (ගියෙහි, යන්නෙහි) and the locative case
% suffix on nouns (සමයෙහි "in that time", ලෝකයෙහි "in the world"). -හු is both
% the 2pl verb ending (ගියෙහු) and a plural nominative on nouns — the grammar's
% own p.100 example "සිත්තරු/සිත්තරුවෝ/සිත්තරහු සිතුවම් අඳිති" has සිත්තරහු as
% the SUBJECT, not the predicate.
%
% Both readings share the same surface form (-යෙහි in each case), so no
% orthographic rule separates them; telling them apart needs part-of-speech
% knowledge this rule base does not have. The 2nd person is also rare in modern
% written prose, where the locative is common, so on real corpora these two
% endings split more often than they should.
%
% Left in place because they are what the grammar specifies, and because
% over-splitting is recoverable — the agglomerative merge can rejoin two
% fragments, whereas a boundary never proposed can never be found. Removing
% them is a one-line change if precision matters more than recall for a given
% corpus.

% Honorific predicate, p.99: "බුදුන් වහන්සේ දනට වඩිති./වඩින සේක."
sentence_word('සේක').

% කෘදන්ත ආඛ්‍යාතය, p.95: a participle plus the particle ය, which stands as its
% own token — "සිරිපාල ගමට යන්නේ ය." / "දැරිය පාසල් ගියා ය." Unlike the pure
% predicate this DOES inflect for gender, but for boundary detection the bare
% particle is the signal.
sentence_word('ය').

% Obligation and evaluative particles that close a sentence, p.107:
% "ඔවුන් එසේ කටයුතු කිරීම වටී." / "කාන්තාවන් විනීත වීම මැනවි."
sentence_word('වටී').
sentence_word('යුතු').
sentence_word('යෙහෙකි').
sentence_word('මැනවි').
sentence_word('හොබී').

% Orthographic terminators.
punctuation('.').
punctuation('?').
punctuation('!').
punctuation('|').    % danda, used for a full stop in older typesetting
punctuation('||').   % double danda

% ===================================================================
% TIER 2 - Clause boundaries: non-finite forms. Sentence CONTINUES.
% ===================================================================

% පූර්වක්‍රියා (absolutive / conjunctive participle) and friends.
%   ලා    absolutive        - කරලා, ගිහිල්ලා
%   මින්  continuous        - කරමින්, බලමින්
%   ගොස්  absolutive of motion
clause_ending('ලා').
clause_ending('මින්').
clause_ending('ගොස්').

% Conditional, p.107: "සෙබළුන් පැමිණියොත් වැසියෝ සොම්නස් වෙති."
%                     "දැරියන් රඟතොත් පෙරහැර විසිතුරු වනු ඇත."
clause_ending('ොත්').
clause_ending('තොත්').

% Temporal, p.107: "අප උගන්වද්දී ළමයෙක් ක්ලාන්ත විය."
clause_ending('ද්දී').
clause_ending('ද්දි').

% Concessive, p.107: "ඔවුන් එහි යතත් අපි නො යමු."
clause_ending('තත්').

% Conditional / topic particle standing alone.
clause_word('නම්').
clause_word('විට').
clause_word('පසු').

% ගොස් is a whole word, not a suffix on a stem, so the suffix rule's length
% guard (which stops the pronoun 'මු' matching the 1pl verb ending) would
% reject it. It needs to be listed here as well.
clause_word('ගොස්').

% ===================================================================
% TIER 3 - Discourse connectives: start a new thought
% ===================================================================

discourse_connective('නමුත්').
discourse_connective('එහෙත්').
discourse_connective('එබැවින්').
discourse_connective('එමනිසා').
discourse_connective('නිසා').
discourse_connective('එනිසා').
discourse_connective('එසේහෙයින්').
discourse_connective('එහෙයින්').

% ===================================================================
% TIER 4 - Never split: particles that bind WITHIN one clause
% ===================================================================

% සහාර්ථය (comitative), p.112: "පියා දරුවන් සමඟ/සහ/කැටුව/හා වැඩ කරයි."
% Splitting at සමඟ orphans the verb කරයි from its subject පියා.
never_split_word('සහ').
never_split_word('සමඟ').
never_split_word('සමග').
never_split_word('හා').
never_split_word('කැටුව').

% සමුච්චයාර්ථය (conjunctive ද), p.110: "මල්ලී ද තෝ ද මම ද එහි යමු."
% ද joins subjects; only the final යමු closes the sentence.
never_split_word('ද').

% විකල්පාර්ථය (disjunctive), p.112: "අපි හෝ ඔවුහු එහි යති."
never_split_word('හෝ').
never_split_word('නොහොත්').

% Negation particle - binds to the verb that follows it.
never_split_word('නො').
never_split_word('නොව').

% ------------------------------------------------------------------
% Pronouns (සර්ව නාම), from the උක්ත/අනුක්ත table on p.91
% ------------------------------------------------------------------
% These must be listed explicitly because several of them END IN A FINITE VERB
% ENDING and would otherwise be read as predicates:
%
%     ඔහු, ඔවුහු, මොවුහු   end in හු  (the 2pl verb ending)
%     අපි, තොපි            end in පි, but ඇතැම්හු / යුෂ්මත්හු end in හු
%     තුමූ                  ends in මූ
%
% "ඔහු පාසල් ගියේ ය." would split after ඔහු, severing the subject from its own
% predicate — the single most damaging false positive in the whole rule base,
% because ඔහු is among the commonest words in Sinhala prose.
%
% The justification is structural, not a patch: Sinhala is SOV and the sentence
% ends with its ආඛ්‍යාතය (p.89), so a pronoun — which is by definition an උක්ත
% or අනුක්ත, never a predicate — cannot be a sentence end. The grammar gives
% this as a closed list, so enumerating it is exactly as principled as
% enumerating the verb endings.

pronoun('මම').  pronoun('මා').  pronoun('අපි').  pronoun('අප').
pronoun('තෝ').  pronoun('තා').  pronoun('තී').
pronoun('ඔබ').  pronoun('නුඹ').  pronoun('නුඹලා').
pronoun('තොපි').  pronoun('තෙපි').  pronoun('තොප').  pronoun('තෙප').
pronoun('යුෂ්මතා').  pronoun('යුෂ්මතී').  pronoun('යුෂ්මත්හු').
pronoun('යුෂ්මතුන්').
pronoun('හේ').  pronoun('ඕ').  pronoun('ඈ').
pronoun('හෙතෙම').  pronoun('තෙමේ').  pronoun('තොමෝ').
pronoun('මෑ').  pronoun('මෝ').
pronoun('ඔහු').  pronoun('ඇය').  pronoun('මැය').
pronoun('ඔවුහු').  pronoun('මොවුහු').  pronoun('තුමූ').
pronoun('ඔවුන්').  pronoun('මොවුන්').  pronoun('මෙවුන්').

% නිත්‍ය බහු වචන, p.92 — likewise subjects, and ඇතැම්හු ends in හු.
pronoun('ඇතැමෙක්').  pronoun('කෙනෙක්').  pronoun('අයෙක්').
pronoun('ඇතැම්හු').  pronoun('ඇතැම්මු').  pronoun('කවුරු').
pronoun('ඇතැමෙකු').  pronoun('කෙනෙකු').  pronoun('අයෙකු').
pronoun('ඇතැමුන්').  pronoun('කවුරුන්').

% ------------------------------------------------------------------
% Deictic locatives (නිපාත / අව්‍යය)
% ------------------------------------------------------------------
% -හි is the locative case suffix as well as the 2sg verb ending, and these
% three are the highest-frequency victims: "මල්ලී ද තෝ ද මම ද එහි යමු." (p.110)
% would split after එහි, cutting the sentence one word before its actual verb.
% They are indeclinables, so like pronouns they can never be a predicate.
%
% This does NOT solve the -හි ambiguity in general. An open-class locative noun
% — සමයෙහි "in that time", ලෝකයෙහි "in the world" — is indistinguishable from a
% 2sg verb on surface form alone, and separating them needs part-of-speech
% information this rule base does not have. Closing the highest-frequency cases
% is what a word list can honestly achieve; the rest is documented as a known
% limitation rather than papered over.

indeclinable('එහි').
indeclinable('මෙහි').
indeclinable('කොහි').

% ===================================================================
% Quotative යි - the hard case (p.103, p.104)
% ===================================================================
%
% යි is ambiguous. It is both:
%   (a) the 3sg finite ending          - කරයි, යයි, වැටෙයි          -> sentence end
%   (b) the quotative closing an        - "සතුරන් ගමට එති'යි ඔවුහු   -> NEVER split
%       embedded clause (අන්තර් වාක්‍යය)    බිය වූහ."
%
% Splitting at (b) severs the sub-clause from the main clause it belongs to,
% producing a chunk with no main verb and a chunk with no subject.
%
% The grammar gives the discriminator (p.104): the quotative attaches to an
% ALREADY FINITE form. So a word ending in <finite ending> + යි is quotative:
%     එති + යි      දෙසති + යි      කීහ + යි      ගනිමි + යි
% whereas a bare stem + යි is the finite 3sg:
%     කර + යි       වැටෙ + යි
%
% Hence: strip the trailing යි and ask whether what remains is itself finite.

is_quotative(Word) :-
    sub_atom(Word, Before, _, 0, 'යි'),
    sub_atom(Word, 0, Before, _, Stem),
    Stem \== '',
    finite_ending(Ending),
    sub_atom(Stem, _, _, 0, Ending),
    !.

% Some typesetting marks the quotative with an apostrophe: එති'යි
is_quotative(Word) :-
    sub_atom(Word, _, _, 0, '\'යි'),
    !.
is_quotative(Word) :-
    sub_atom(Word, _, _, 0, '’යි'),
    !.

% ===================================================================
% Inference rules
% ===================================================================

% Never split takes precedence over everything: these are the words whose
% splitting is known to destroy meaning, so no later rule may override them.
is_never_split(Word) :-
    never_split_word(Word), !.
is_never_split(Word) :-
    pronoun(Word), !.
is_never_split(Word) :-
    indeclinable(Word), !.
is_never_split(Word) :-
    is_quotative(Word), !.

% --- sentence level ---

is_sentence_end(Word) :-
    is_never_split(Word), !, fail.

% Terminal punctuation anywhere at the end of the token.
is_sentence_end(Word) :-
    punctuation(Punct),
    sub_atom(Word, _, _, 0, Punct), !.

% A standalone sentence-final word (ය, වටී, මැනවි, සේක ...).
is_sentence_end(Word) :-
    sentence_word(Word), !.

% A finite verb ending. Guarded on length so that a two-character word which
% merely *is* the ending (e.g. the pronoun 'මු') is not mistaken for a verb
% carrying it.
is_sentence_end(Word) :-
    finite_ending(Ending),
    sub_atom(Word, _, _, 0, Ending),
    atom_length(Word, Length),
    atom_length(Ending, EndingLength),
    Length > EndingLength,
    !.

% --- clause level ---
% Every sentence end is also a clause end: a sentence boundary is a fortiori a
% clause boundary, so the clause pass is a superset of the sentence pass.

is_clause_end(Word) :-
    is_never_split(Word), !, fail.

is_clause_end(Word) :-
    is_sentence_end(Word), !.

is_clause_end(Word) :-
    clause_word(Word), !.

is_clause_end(Word) :-
    discourse_connective(Word), !.

is_clause_end(Word) :-
    clause_ending(Ending),
    sub_atom(Word, _, _, 0, Ending),
    atom_length(Word, Length),
    atom_length(Ending, EndingLength),
    Length > EndingLength,
    !.

% ===================================================================
% Python-facing interface
% ===================================================================
%
% Total in both arities: the host must never observe a Prolog failure during a
% routine query, because a failure is indistinguishable from a transport error
% on the MQI socket.

check_split(Word, sentence, 'true') :- is_sentence_end(Word), !.
check_split(Word, clause,   'true') :- is_clause_end(Word), !.
check_split(_, _, 'false').

% Section 6.5.2's original arity, kept so the prototype's caller still works.
check_split(Word, Result) :- check_split(Word, sentence, Result).
