import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Container from '@mui/material/Container';
import InputLabel from '@mui/material/InputLabel';
import Link from '@mui/material/Link';
import Stack from '@mui/material/Stack';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid2';
import { visuallyHidden } from '@mui/utils';
import { styled } from '@mui/material/styles';

import Divider from '@mui/material/Divider';

const StyledBox = styled('div')(({ theme }) => ({
  alignSelf: 'center',
  width: '100%',
  height: 400,
  marginTop: theme.spacing(8),
  borderRadius: (theme).shape.borderRadius,
  outline: '6px solid',
  outlineColor: 'hsla(220, 25%, 80%, 0.2)',
  border: '1px solid',
  borderColor: (theme).palette.grey[200],
  boxShadow: '0 0 12px 8px hsla(220, 25%, 80%, 0.2)',
  backgroundImage: `url(https://mui.com/static/screenshots/material-ui/getting-started/templates/dashboard.jpg)`,
  backgroundSize: 'cover',
  [theme.breakpoints.up('sm')]: {
    marginTop: theme.spacing(10),
    height: 700,
  },
  ...theme.applyStyles('dark', {
    boxShadow: '0 0 24px 12px hsla(210, 100%, 25%, 0.2)',
    backgroundImage: `url(https://mui.com/static/screenshots/material-ui/getting-started/templates/dashboard-dark.jpg)`,
    outlineColor: 'hsla(220, 20%, 42%, 0.1)',
    borderColor: (theme).palette.grey[700],
  }),
}));

export default function Hero() {
  return (
    <Box
      id="hero"
      sx={(theme) => ({
        width: '100%',
        backgroundRepeat: 'no-repeat',

        backgroundImage:
          'radial-gradient(ellipse 80% 50% at 50% -20%, hsl(210, 100%, 90%), transparent)',
        ...theme.applyStyles('dark', {
          backgroundImage:
            'radial-gradient(ellipse 80% 50% at 50% -20%, hsl(210, 100%, 16%), transparent)',
        }),
      })}
    >
      <Container
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          pt: { xs: 14, sm: 20 },
          pb: { xs: 8, sm: 12 },
        }}
      >
        <Stack
          spacing={2}
          useFlexGap
          sx={{ alignItems: 'center', width: { xs: '100%', sm: '70%' } }}
        >
          <Typography
            variant="h3"
            sx={{
              display: 'flex',
              flexDirection: { xs: 'column', sm: 'row' },
              alignItems: 'center',
              fontSize: 'clamp(3rem, 10vw, 3.5rem)',
            }}
          >
            Dhwani
          </Typography>

          <Typography
            sx={{
              textAlign: 'center',
              color: 'text.secondary',
              width: { sm: '100%', md: '80%' },
            }}
          >
          Your Kannada-Speaking Voice Buddy
          </Typography>
          <Typography
            sx={{
              textAlign: 'center',
              color: 'text.secondary',
              width: { sm: '100%', md: '80%' },
            }}
          >
            Imagine chatting with your phone in Kannada—Dhwani makes it happen!
          
          </Typography>
          
          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            spacing={1}
            useFlexGap
            sx={{ pt: 2, width: { xs: '100%', sm: '350px' } }}
          >
            <div >
          <Grid container spacing={2}>
            <Grid size={{ xs: 12 }} >
            <iframe
              width="560"
              height="315"
              src="https://www.youtube.com/embed/sxQp-zPJBO8?rel=0"
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
              title="Watch the video"
            ></iframe>
            </Grid>
          </Grid>
          </div>

          <Divider />

            <div style={{ display: 'none' }}>
            <InputLabel htmlFor="email-hero" sx={visuallyHidden}>
              Email
            </InputLabel>
            <TextField
              id="email-hero"
              hiddenLabel
              size="small"
              variant="outlined"
              aria-label="Enter your email address"
              placeholder="Your email address"
              fullWidth
              slotProps={{
                htmlInput: {
                  autoComplete: 'off',
                  'aria-label': 'Enter your email address',
                },
              }}
            />
            <Button
              variant="contained"
              color="primary"
              size="small"
              sx={{ minWidth: 'fit-content' }}
            >
              Start now
            </Button>
            </div>
          </Stack>

          <div style={{ display: 'none' }}>
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ textAlign: 'center' }}
          >
            By clicking &quot;Start now&quot; you agree to our&nbsp;
            <Link href="#" color="primary">
              Terms & Conditions
            </Link>
            .
          </Typography>
          </div>
          
        </Stack>


        <Stack
          spacing={2}
          useFlexGap
          sx={{ alignItems: 'center', width: { xs: '100%', sm: '70%' } }}
        >

        <Divider />

        <Divider />
        <Divider />

<Typography
  sx={{
    textAlign: 'center',
    color: 'text.secondary',
    width: { sm: '100%', md: '80%' },
  }}
>
   Now Support 5 languages on Android. 
   Email Us - TO get access to early user program
</Typography>

<div >
<Grid container spacing={2}>
  <Grid size={{ xs: 12 }} >
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/31jvakEM_TM?rel=0"
    frameBorder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowFullScreen
    title="Watch the video"
  ></iframe>
  </Grid>
</Grid>
</div>

</Stack>
<Stack
          spacing={2}
          useFlexGap
          sx={{ alignItems: 'center', width: { xs: '100%', sm: '70%' } }}
        >

        <Divider />

        <Divider />
        <Divider />

<Typography
  sx={{
    textAlign: 'center',
    color: 'text.secondary',
    width: { sm: '100%', md: '80%' },
  }}
>
Introduction to Dhwani
</Typography>

<div >
<Grid container spacing={2}>
  <Grid size={{ xs: 12 }} >
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/kqZZZjbeNVk?rel=0"
    frameBorder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowFullScreen
    title="Watch the video"
  ></iframe>
  </Grid>
</Grid>
</div>

</Stack>

<Stack
          spacing={2}
          useFlexGap
          sx={{ alignItems: 'center', width: { xs: '100%', sm: '70%' } }}
        >

        <Divider />

        <Divider />
        <Divider />

<Typography
  sx={{
    textAlign: 'center',
    color: 'text.secondary',
    width: { sm: '100%', md: '80%' },
  }}
>
Access to Dhwani AI - API
</Typography>

<div >
<Grid container spacing={2}>
  <Grid size={{ xs: 12 }} >
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/RLIhG1bt8gw?rel=0"
    frameBorder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowFullScreen
    title="Watch the video"
  ></iframe>
  </Grid>
</Grid>
</div>

</Stack>





        <div style={{ display: 'none' }}>
        <StyledBox id="image" />
        </div>
      </Container>
    </Box>
  );
}
