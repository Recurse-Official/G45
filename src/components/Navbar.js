import React from 'react';
import styled from 'styled-components';
import { Link } from 'react-router-dom';
const Nav = styled.nav`
  background: var(--color-secondary);
  height: 70px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 2px 5px var(--color-box-shadow);
  position: fixed;
  width: 100%;
  top: 0;
  z-index: 1000;
`;

const NavLinksContainer = styled.div`
  display: flex;
  flex: 1;
  justify-content: center;
  align-items: center;
`;

const NavLinks = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
`;

const NavLink = styled(Link)`
  color: var(--color-text);
  text-decoration: none;
  padding: 0 1rem;
  font-size: 1.2rem;
  transition: color 0.3s;

  &:hover {
    color: var(--color-primary);
  }
`;

const Navbar = () => {
  return (
    <Nav>
      <NavLinksContainer>
        <NavLinks>
          <NavLink to="/qandagenerator">Q&A Generator</NavLink>
          <NavLink to="/QuestionAnswering">QuestionAnswering</NavLink>
        </NavLinks>
      </NavLinksContainer>
    </Nav>
  );
};

export default Navbar;
